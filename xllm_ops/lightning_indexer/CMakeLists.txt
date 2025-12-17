add_ops_compile_options(
	OP_NAME LightningIndexer
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -DASCENDC_DUMP=0
)

target_sources(op_host_aclnn PRIVATE
        lightning_indexer_cpu.cpp
)

# optiling
target_sources(optiling PRIVATE
        lightning_indexer_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common
)


# opproto
target_sources(opsproto PRIVATE
        lightning_indexer_proto.cpp
)

target_include_directories(opsproto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
install(FILES lightning_indexer.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES lightning_indexer_kernel.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
