import torch
import custom_ops_lib


def replace_token(self, other):
    return custom_ops_lib.replace_token(self, other)
