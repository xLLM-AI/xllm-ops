# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
"""Check DSA kernel include paths and packaging without compiling device code."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SHARED_OP = "dsa_kv_offloading_common"
HEADER = "dsa_kv_offloading_plan_serve.h"
SHARED_KERNEL = ROOT / "xllm_ops" / SHARED_OP / "op_kernel"
OPS = {
    "dsa_kv_offloading_plan_nope": "DsaKvOffloadingPlanNope",
    "dsa_kv_offloading_plan_rope": "DsaKvOffloadingPlanRope",
    "dsa_kv_offloading_serve_nope": "DsaKvOffloadingServeNope",
    "dsa_kv_offloading_serve_rope": "DsaKvOffloadingServeRope",
}


def run(*args: str) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module", params=["Unix Makefiles", "Ninja"])
def kernel_build(tmp_path_factory, request):
    tool = "ninja" if request.param == "Ninja" else "make"
    if not all(shutil.which(name) for name in ["cmake", "g++", tool]):
        pytest.skip(f"cmake, g++, and {tool} are required")
    work = tmp_path_factory.mktemp("dsa-kernel-build")
    source = work / "source"
    source.mkdir()
    build = work / "build"
    # Reproduce a retained build cache whose global common directory is empty.
    (build / "src/ascendc/common").mkdir(parents=True)
    (source / "CMakeLists.txt").write_text(
        f"""cmake_minimum_required(VERSION 3.16)
project(dsa_kernel_build NONE)
include("{ROOT.as_posix()}/cmake/func.cmake")
set(BUILD_OPEN_PROJECT OFF)
set(BUILD_OPS_RTY_KERNEL ON)
set(ADD_OPS_COMPILE_OPTION_V2 ON)
set(HI_PYTHON "{sys.executable}")
set(ASCENDC_CMAKE_UTIL_DIR "{ROOT.as_posix()}/cmake/scripts/util")
set(ASCEND_CUSTOM_OPTIONS "${{CMAKE_BINARY_DIR}}/autogen/options.ini")
set(OPS_ADV_UTILS_KERNEL_INC "{ROOT.as_posix()}/common/include/kernel")
set(IMPL_INSTALL_DIR impl)
file(MAKE_DIRECTORY "${{CMAKE_BINARY_DIR}}/autogen")
file(WRITE "${{CMAKE_BINARY_DIR}}/dependencies.txt" "")
add_custom_target(stage)
foreach(op IN ITEMS {' '.join(OPS)})
  add_subdirectory("{ROOT.as_posix()}/xllm_ops/${{op}}/op_host" "${{op}}")
  file(APPEND "${{CMAKE_BINARY_DIR}}/dependencies.txt" "${{op}},${{${{op}}_depends}}\n")
  add_ops_src_copy(TARGET_NAME ${{op}}_copy
    SRC "{ROOT.as_posix()}/xllm_ops/${{op}}"
    DST "${{CMAKE_BINARY_DIR}}/src/${{op}}" COMPUTE_UNIT ascend910_93)
  foreach(dependency ${{${{op}}_depends}})
    add_ops_src_copy(TARGET_NAME ${{dependency}}_copy
      SRC "{ROOT.as_posix()}/xllm_ops/${{dependency}}"
      DST "${{CMAKE_BINARY_DIR}}/src/${{dependency}}"
      COMPUTE_UNIT ascend910_93 BE_RELIED ${{op}}_copy)
  endforeach()
  add_dependencies(stage ${{op}}_copy)
  # Fixture for the unchanged package layout of each kernel source.
  install(FILES "{ROOT.as_posix()}/xllm_ops/${{op}}/op_kernel/${{op}}.cpp"
    DESTINATION "${{IMPL_INSTALL_DIR}}/ascendc/${{op}}")
endforeach()
"""
    )
    run(
        "cmake", "-S", str(source), "-B", str(build), "-G", request.param,
        f"-DCMAKE_MAKE_PROGRAM={shutil.which(tool)}",
    )
    run("cmake", "--build", str(build), "--target", "stage", "-j", "2")
    package = work / "relocated-package"
    run("cmake", "--install", str(build), "--prefix", str(package))
    # Only preprocess: AscendC types/instructions need no host implementation.
    stubs = work / "stubs"
    stubs.mkdir()
    (stubs / "kernel_operator.h").write_text("// device declarations omitted\n")
    options = {}
    for line in (build / "autogen/options.ini").read_text().splitlines():
        op_type, _, flags = line.split(",", 2)
        options[op_type] = flags.split(";")
    return build, package, stubs, options


def test_header_is_owned_by_dsa() -> None:
    assert (SHARED_KERNEL / HEADER).is_file()
    assert not (ROOT / "common/include/kernel" / HEADER).exists()


def test_shared_dependency_is_staged(kernel_build) -> None:
    build, _, _, _ = kernel_build
    assert (build / "dependencies.txt").read_text().splitlines() == [
        f"{op},{SHARED_OP}" for op in OPS
    ]
    assert (build / "src" / SHARED_OP / "op_kernel" / HEADER).read_bytes() == (
        SHARED_KERNEL / HEADER
    ).read_bytes()
    assert not (build / "src/ascendc/common" / HEADER).exists()


@pytest.mark.parametrize("op", OPS)
def test_kernel_uses_explicit_shared_include(kernel_build, op: str) -> None:
    build, _, stubs, options = kernel_build
    flags = [flag for flag in options[OPS[op]] if flag.startswith("-I")]
    assert f"-I{SHARED_KERNEL}" in flags
    output = run(
        "g++", "-E", f"-I{stubs}", *flags,
        str(build / "src" / op / "op_kernel" / f"{op}.cpp"),
    )
    assert str(SHARED_KERNEL / HEADER) in output


@pytest.mark.parametrize("op", OPS)
def test_installed_kernel_needs_no_source_include(kernel_build, op: str) -> None:
    _, package, stubs, _ = kernel_build
    kernel = package / "impl/ascendc" / op
    assert (kernel / HEADER).read_bytes() == (SHARED_KERNEL / HEADER).read_bytes()
    output = run("g++", "-E", f"-I{stubs}", str(kernel / f"{op}.cpp"))
    assert str(kernel / HEADER) in output
