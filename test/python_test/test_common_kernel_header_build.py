# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
"""Exercise public kernel header copying without CANN or device compilation."""

import shutil
import subprocess
from pathlib import Path

import pytest


FUNC_CMAKE = Path(__file__).resolve().parents[2] / "cmake/func.cmake"
DSA_HEADER = "dsa_kv_offloading_plan_serve.h"


def run(*args: str) -> None:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def build(build_dir: Path) -> None:
    run("cmake", "--build", str(build_dir), "--target", "header_copy", "-j", "2")


@pytest.fixture(params=["Unix Makefiles", "Ninja"])
def header_build(tmp_path: Path, request: pytest.FixtureRequest):
    tool = "ninja" if request.param == "Ninja" else "make"
    if not shutil.which("cmake") or not shutil.which(tool):
        pytest.skip(f"cmake and {tool} are required")

    source = tmp_path / "source"
    headers = source / "headers"
    headers.mkdir(parents=True)
    (headers / "existing.h").write_text("// existing header\n")
    op = source / "op"
    op.mkdir()
    (op / "kernel.cpp").write_text("// kernel source\n")
    (source / "CMakeLists.txt").write_text(
        f"""cmake_minimum_required(VERSION 3.16)
project(common_kernel_header_build NONE)
include("{FUNC_CMAKE.as_posix()}")
set(OPS_ADV_UTILS_KERNEL_INC "${{CMAKE_CURRENT_SOURCE_DIR}}/headers")
add_ops_src_copy(
    TARGET_NAME header_copy
    SRC "${{CMAKE_CURRENT_SOURCE_DIR}}/op"
    DST "${{CMAKE_BINARY_DIR}}/src/op"
    COMPUTE_UNIT ascend910_93)
"""
    )
    build_dir = tmp_path / "build"
    run(
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build_dir),
        "-G",
        request.param,
        f"-DCMAKE_MAKE_PROGRAM={shutil.which(tool)}",
    )
    return headers, build_dir / "src/ascendc/common", build_dir


def test_fresh_build_copies_headers(header_build) -> None:
    headers, copied, build_dir = header_build
    build(build_dir)
    assert (copied / "existing.h").read_bytes() == (
        headers / "existing.h"
    ).read_bytes()


def test_existing_directory_does_not_skip_copy(header_build) -> None:
    headers, copied, build_dir = header_build
    (headers / DSA_HEADER).write_text("// DSA header\n")
    copied.mkdir(parents=True)
    build(build_dir)
    assert (copied / DSA_HEADER).read_bytes() == (headers / DSA_HEADER).read_bytes()


def test_new_header_is_copied_without_manual_configure(header_build) -> None:
    headers, copied, build_dir = header_build
    build(build_dir)
    (headers / DSA_HEADER).write_text("// newly added DSA header\n")
    build(build_dir)
    assert (copied / DSA_HEADER).read_bytes() == (headers / DSA_HEADER).read_bytes()


def test_changed_header_is_updated(header_build) -> None:
    headers, copied, build_dir = header_build
    build(build_dir)
    (headers / "existing.h").write_text("// changed header\n")
    build(build_dir)
    assert (copied / "existing.h").read_bytes() == (
        headers / "existing.h"
    ).read_bytes()


def test_missing_copied_header_is_restored(header_build) -> None:
    _, copied, build_dir = header_build
    build(build_dir)
    (copied / "existing.h").unlink()
    build(build_dir)
    assert (copied / "existing.h").is_file()


def test_unchanged_build_does_not_copy_again(header_build) -> None:
    _, copied, build_dir = header_build
    build(build_dir)
    header = copied / "existing.h"
    before = header.stat().st_mtime_ns
    build(build_dir)
    assert header.stat().st_mtime_ns == before
