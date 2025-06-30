import os
import sys
import numpy as np

loss = 1e-4


def verify_result(real_result, golden):
    result = np.fromfile(real_result, dtype=np.float16)
    golden = np.fromfile(golden, dtype=np.float16)
    # 打印result和golden ，到小数点后3位，矩阵要全部打印出来不能省略
    np.set_printoptions(precision=3, suppress=False, linewidth=200, threshold=np.inf)
    # 打印前32个数
    print("result\n")
    print(result[:32])
    print("golden\n")
    print(golden[:32])
    for i in range(len(result)):
        diff = abs(result[i] - golden[i])
        if (diff > loss) and (diff / golden[i] > loss):
            error_message = "output[{}] is {}, expect {}".format(i, result[i], golden[i])
            print(error_message)
            return False
    print("\ntest pass")
    return True


if __name__ == '__main__':
    verify_result(sys.argv[1], sys.argv[2])
