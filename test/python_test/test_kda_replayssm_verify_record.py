# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator

import pytest
import torch

from test_kda_replayssm_correction_fold import _bind, _check

_POINTER = ctypes.c_void_p
_SHAPE = ctypes.POINTER(ctypes.c_int64)
_UINT64 = ctypes.c_uint64
_DTYPES = {torch.float32: 0, torch.int32: 3, torch.bfloat16: 27}
_SCALE = 128**-0.5


class _VerifyRecord:
    def __init__(self, library: Path) -> None:
        ops = ctypes.CDLL(str(library))
        base = ctypes.CDLL("libopapi.so")
        self.create_tensor = _bind(
            base,
            "aclCreateTensor",
            _POINTER,
            _SHAPE,
            _UINT64,
            ctypes.c_int,
            _SHAPE,
            ctypes.c_int64,
            ctypes.c_int,
            _SHAPE,
            _UINT64,
            _POINTER,
        )
        self.destroy_tensor = _bind(base, "aclDestroyTensor", ctypes.c_int, _POINTER)
        self.get_workspace = _bind(
            ops,
            "aclnnKdaReplayssmVerifyRecordGetWorkspaceSize",
            ctypes.c_int,
            *([_POINTER] * 9),
            ctypes.c_double,
            ctypes.c_int64,
            *([_POINTER] * 4),
            ctypes.POINTER(_UINT64),
            ctypes.POINTER(_POINTER),
        )
        self.run_op = _bind(
            ops,
            "aclnnKdaReplayssmVerifyRecord",
            ctypes.c_int,
            _POINTER,
            _UINT64,
            _POINTER,
            _POINTER,
        )

    def _tensor(self, tensor: torch.Tensor) -> int:
        shape = (ctypes.c_int64 * tensor.ndim)(*tensor.shape)
        stride = (ctypes.c_int64 * tensor.ndim)(*tensor.stride())
        storage = tensor.untyped_storage()
        storage_shape = (ctypes.c_int64 * 1)(storage.nbytes() // tensor.element_size())
        handle = self.create_tensor(
            shape,
            tensor.ndim,
            _DTYPES[tensor.dtype],
            stride,
            tensor.storage_offset(),
            2,
            storage_shape,
            1,
            storage.data_ptr(),
        )
        assert handle, "could not create an ACL tensor descriptor"
        return handle

    @contextmanager
    def prepare(
        self,
        inputs: tuple[torch.Tensor, ...],
        outputs: tuple[torch.Tensor, ...],
    ) -> Iterator[tuple[torch.Tensor, _UINT64, _POINTER]]:
        with ExitStack() as cleanup:
            handles = []
            for tensor in inputs + outputs:
                handle = self._tensor(tensor)
                cleanup.callback(self.destroy_tensor, handle)
                handles.append(handle)
            size, executor = _UINT64(), _POINTER()
            _check(
                self.get_workspace(
                    *handles[:9],
                    _SCALE,
                    inputs[0].stride(0),
                    *handles[9:],
                    ctypes.byref(size),
                    ctypes.byref(executor),
                )
            )
            workspace = torch.empty(
                size.value, dtype=torch.uint8, device=inputs[0].device
            )
            yield workspace, size, executor

    def launch(self, prepared: tuple[torch.Tensor, _UINT64, _POINTER]) -> None:
        workspace, size, executor = prepared
        _check(
            self.run_op(
                workspace.data_ptr(),
                size,
                executor,
                torch.npu.current_stream().npu_stream,
            )
        )


def test_verify_record_switches_qkv_layouts_in_one_process() -> None:
    library = os.environ.get("XLLM_REPLAYSSM_ACLNN_LIBRARY")
    device_name = os.environ.get("XLLM_KDA_TEST_NPU_DEVICE")
    if not library or not device_name:
        pytest.skip("requires the isolated ReplaySSM vendor and an explicit NPU")
    torch.npu.set_device(device_name)
    device = torch.device(device_name)
    verify = _VerifyRecord(Path(library))
    generator = torch.Generator().manual_seed(61)

    def random_tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.randn(shape, generator=generator).to(dtype).to(device)

    packed = random_tensor((8, 3, 8, 128), torch.bfloat16)
    packed_qkv = packed.unbind(dim=1)
    dense_qkv = tuple(view.contiguous() for view in packed_qkv)
    assert all(view.stride(0) == 3072 for view in packed_qkv)
    assert all(view.stride(0) == 1024 for view in dense_qkv)

    gate = random_tensor((8, 8, 128), torch.bfloat16)
    beta = random_tensor((8, 8), torch.bfloat16)
    state = random_tensor((3, 8, 128, 128), torch.float32).mul_(0.2)
    a_log = random_tensor((8,), torch.float32)
    dt_bias = random_tensor((1024,), torch.float32)
    indices = torch.zeros(1, device=device, dtype=torch.int32)
    metadata = (gate, beta, state, a_log, dt_bias, indices)
    expected = (
        *(
            torch.full((3, 8, 8, 128), -12.0, dtype=torch.float32, device=device)
            for _ in range(3)
        ),
        torch.full_like(gate, -12),
    )
    actual = tuple(tensor.clone() for tensor in expected)

    stream = torch.npu.Stream(device=device)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        with verify.prepare(dense_qkv + metadata, expected) as eager:
            verify.launch(eager)
            stream.synchronize()
        with verify.prepare(packed_qkv + metadata, actual) as prepared:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                verify.launch(prepared)
            indices.fill_(2)
            for records in expected[:3]:
                records.fill_(-12.0)
            with verify.prepare(dense_qkv + metadata, expected) as eager:
                verify.launch(eager)
            graph.replay()
            stream.synchronize()
            graph.reset()
        assert expected[-1].count_nonzero().item() > 0
        for packed_result, dense_result in zip(actual, expected):
            dtype = (
                torch.int16 if packed_result.dtype == torch.bfloat16 else torch.int32
            )
            assert torch.equal(packed_result.view(dtype), dense_result.view(dtype))
