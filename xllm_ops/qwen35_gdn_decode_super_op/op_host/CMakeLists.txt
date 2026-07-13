if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        qwen35_gdn_decode_super_op_def.cpp
    )
endif()

if (DEFINED ENV{TILELANG_ROOT})
    set(QWEN35_TILELANG_ROOT "$ENV{TILELANG_ROOT}")
else()
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
                "import pathlib, tilelang; print(pathlib.Path(tilelang.__file__).resolve().parent)"
        OUTPUT_VARIABLE QWEN35_TILELANG_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE QWEN35_TILELANG_DISCOVERY_RESULT
    )
    if (NOT QWEN35_TILELANG_DISCOVERY_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Qwen35GdnDecodeSuperOp requires TileLang; set TILELANG_ROOT")
    endif()
endif()

get_filename_component(
    QWEN35_CATLASS_INCLUDE
    "${CMAKE_CURRENT_LIST_DIR}/../../../third_party/catlass/include"
    ABSOLUTE
)

add_ops_compile_options(
    OP_NAME Qwen35GdnDecodeSuperOp
    OPTIONS --cce-auto-sync=on
            -I${QWEN35_TILELANG_ROOT}/src
            -I${QWEN35_TILELANG_ROOT}/3rdparty/shmem/include
            -I${QWEN35_TILELANG_ROOT}/3rdparty/shmem/src/device
            -I${QWEN35_CATLASS_INCLUDE}
            -Wno-deprecated-declarations
            -Wno-macro-redefined
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE qwen35_gdn_decode_super_op ACLNNTYPE aclnn)
endif()
