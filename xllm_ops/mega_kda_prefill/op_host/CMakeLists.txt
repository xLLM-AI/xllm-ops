# Copyright 2026 The xLLM Authors. All Rights Reserved.
add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE mega_kda_prefill_def.cpp)
endif()
add_ops_compile_options(
    OP_NAME MegaKdaPrefill
    OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations
            -I${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/pto-isa/include
)
if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE mega_kda_prefill ACLNNTYPE aclnn)
endif()
