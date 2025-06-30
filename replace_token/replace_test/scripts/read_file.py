#read file and print the content
import os
import sys
import numpy as np

out_forked_token_ids = np.fromfile('/export/home/limenxin1/projects/xllm_ops/replace_token/replace_test/output/output_out.bin', dtype=np.int32)
print(out_forked_token_ids)
print(out_forked_token_ids.shape)
in_forked_token_ids = np.fromfile('/export/home/limenxin1/projects/xllm_ops/replace_token/replace_test/input/output_out.bin', dtype=np.int32)
print(in_forked_token_ids)
print(in_forked_token_ids.shape)












