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
from typing import Any, Callable, Iterator

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

_POINTER = ctypes.c_void_p
_SHAPE = ctypes.POINTER(ctypes.c_int64)
_UINT64 = ctypes.c_uint64
_ACL_DTYPES = {torch.float32: 0, torch.int32: 3}


def _bind(
    library: ctypes.CDLL, name: str, result: object, *args: object
) -> Callable[..., Any]:
    function = getattr(library, name)
    function.restype = result
    function.argtypes = args
    return function


def _check(status: int) -> None:
    if status != 0:
        get_recent_error = _bind(
            ctypes.CDLL("libascendcl.so"), "aclGetRecentErrMsg", ctypes.c_char_p
        )
        raise RuntimeError(f"ACLNN status {status}: {get_recent_error()!r}")


class _Fold:
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
        self.create_list = _bind(
            base,
            "aclCreateTensorList",
            _POINTER,
            ctypes.POINTER(_POINTER),
            _UINT64,
        )
        self.destroy_list = _bind(base, "aclDestroyTensorList", ctypes.c_int, _POINTER)
        self.get_workspace = _bind(
            ops,
            "aclnnKdaReplayssmCorrectionFoldGetWorkspaceSize",
            ctypes.c_int,
            *([_POINTER] * 7),
            ctypes.POINTER(_UINT64),
            ctypes.POINTER(_POINTER),
        )
        self.run_op = _bind(
            ops,
            "aclnnKdaReplayssmCorrectionFold",
            ctypes.c_int,
            _POINTER,
            _UINT64,
            _POINTER,
            _POINTER,
        )

    def _tensor(self, tensor: torch.Tensor) -> int:
        assert tensor.is_contiguous()
        shape = (ctypes.c_int64 * tensor.ndim)(*tensor.shape)
        stride = (ctypes.c_int64 * tensor.ndim)(*tensor.stride())
        handle = self.create_tensor(
            shape,
            tensor.ndim,
            _ACL_DTYPES[tensor.dtype],
            stride,
            0,
            2,
            shape,
            tensor.ndim,
            tensor.data_ptr(),
        )
        assert handle, "could not create an ACL tensor descriptor"
        return handle

    def _tensor_list(self, tensors: tuple[torch.Tensor, ...]) -> int:
        descriptors = (_POINTER * len(tensors))(
            *(self._tensor(tensor) for tensor in tensors)
        )
        handle = self.create_list(descriptors, len(tensors))
        assert handle, "could not create an ACL tensor-list descriptor"
        return handle

    @contextmanager
    def prepare(
        self,
        states: tuple[torch.Tensor, ...],
        keys: tuple[torch.Tensor, ...],
        corrections: tuple[torch.Tensor, ...],
        decays: tuple[torch.Tensor, ...],
        slots: torch.Tensor,
        accepted: torch.Tensor,
    ) -> Iterator[tuple[torch.Tensor, _UINT64, _POINTER]]:
        with ExitStack() as cleanup:
            lists = []
            for tensors in (states, keys, corrections, decays):
                handle = self._tensor_list(tensors)
                cleanup.callback(self.destroy_list, handle)
                lists.append(handle)
            indices = self._tensor(slots)
            cleanup.callback(self.destroy_tensor, indices)
            lengths = self._tensor(accepted)
            cleanup.callback(self.destroy_tensor, lengths)
            size = _UINT64()
            executor = _POINTER()
            _check(
                self.get_workspace(
                    *lists,
                    indices,
                    lengths,
                    lists[0],
                    ctypes.byref(size),
                    ctypes.byref(executor),
                )
            )
            workspace = torch.empty(size.value, device=slots.device, dtype=torch.uint8)
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


def _reference(
    before: tuple[torch.Tensor, ...],
    records: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    slots: list[int],
    accepted: list[int],
) -> tuple[torch.Tensor, ...]:
    result = tuple(state.clone() for state in before)
    for state, (keys, corrections, decays) in zip(result, records):
        for slot, count in zip(slots, accepted):
            if 0 <= slot < state.shape[0]:
                for token in range(min(count, 8)):
                    state[slot].mul_(decays[slot, token].unsqueeze(1))
                    state[slot].addcmul_(
                        corrections[slot, token].unsqueeze(-1),
                        keys[slot, token].unsqueeze(-2),
                    )
    return result


@pytest.mark.parametrize("graph_mode", (False, True))
def test_fold_updates_independent_layers_in_place(graph_mode: bool) -> None:
    library = os.environ.get("XLLM_REPLAYSSM_ACLNN_LIBRARY")
    device_name = os.environ.get("XLLM_KDA_TEST_NPU_DEVICE")
    if not library or not device_name:
        pytest.skip("requires the isolated ReplaySSM vendor and an explicit NPU")
    device = torch.device(device_name)
    torch.npu.set_device(device)
    fold = _Fold(Path(library))
    generator = torch.Generator().manual_seed(82)

    def fractions(size: tuple[int, ...]) -> torch.Tensor:
        return torch.randint(-3, 4, size, generator=generator).float().mul_(0.125)

    originals = tuple(fractions((capacity, 8, 128, 128)) for capacity in (3, 5))
    records = tuple(
        (
            fractions((capacity, 8, 8, 128)),
            fractions((capacity, 8, 8, 128)),
            torch.randint(1, 3, (capacity, 8, 8, 128), generator=generator)
            .float()
            .mul_(0.5),
        )
        for capacity in (3, 5)
    )
    states = tuple(state.to(device) for state in originals)
    keys, corrections, decays = tuple(
        zip(*(tuple(tensor.to(device) for tensor in layer) for layer in records))
    )
    assert (
        states[0].untyped_storage().data_ptr() != states[1].untyped_storage().data_ptr()
    )
    addresses = tuple(state.data_ptr() for state in states)
    logical_slots = [-1, 2, 4]
    slots = torch.tensor(logical_slots, dtype=torch.int32, device=device)
    lengths = torch.ones_like(slots)
    arguments = (states, keys, corrections, decays, slots, lengths)
    stream = torch.npu.Stream(device=device)
    stream.wait_stream(torch.npu.current_stream())
    graph = None
    with torch.npu.stream(stream):
        with fold.prepare(*arguments) as warmup:
            fold.launch(warmup)
            stream.synchronize()
        with ExitStack() as cleanup:
            if graph_mode:
                prepared = cleanup.enter_context(fold.prepare(*arguments))
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=stream):
                    fold.launch(prepared)
            for counts in ([2, 7, 3], [2, 0, 9]):
                for target, initial in zip(states, originals):
                    target.copy_(initial, non_blocking=False)
                lengths.copy_(
                    torch.tensor(counts, dtype=torch.int32), non_blocking=False
                )
                if graph is None:
                    with fold.prepare(*arguments) as eager:
                        fold.launch(eager)
                        stream.synchronize()
                else:
                    graph.replay()
                    stream.synchronize()
                expected = _reference(originals, records, logical_slots, counts)
                assert tuple(state.data_ptr() for state in states) == addresses
                for state, reference in zip(states, expected):
                    assert torch.equal(
                        state.cpu().view(torch.int32), reference.view(torch.int32)
                    )
            if graph is not None:
                graph.reset()
