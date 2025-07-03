# python3 verify_result.py /export/home/limenxin1/projects/indexTest/output/output_z.bin /export/home/limenxin1/projects/new_master/IndexGroupMatmul/GroupMatmul/FrameworkLaunch/NativeGroupMatmul/output/output_z.bin
CURRENT=/export/home/limenxin1/projects/xllm_ops/index_group_gemm/test/index_group_gemm_test/scripts
python3 $CURRENT/verify_result.py /export/home/limenxin1/projects/xllm_ops/index_group_gemm/test/index_group_gemm_test/output/output_z.bin \
    /export/home/limenxin1/projects/xllm_ops/index_group_gemm/test/aclnn_grouped_matmul_v4_test/output/output_z.bin