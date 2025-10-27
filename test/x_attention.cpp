// #include <gtest/gtest.h>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "x_attention.h"
#include "base/utils_tensor.h"

using namespace std;

int main() {
    
    xattention::XAttentionBase inputs(2, 512, 8, 1, 128, 256, 1);
    int deviceId = 1;
    aclrtStream stream1;
    utils::initialize_acl(deviceId, &stream1);
    inputs.create_torch_tensor();
    cout << "create tensors done" << endl;
    xattention::XAttentionOp op(inputs);
    cout << "create op done" << endl;
    op.process(stream1);
    cout << "process done" << endl;
    aclrtSynchronizeStream(stream1);
    //hello

    op.destroy_tensors();
    aclrtDestroyStream(stream1);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
