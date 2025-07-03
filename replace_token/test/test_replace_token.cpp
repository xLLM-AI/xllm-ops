#include <iostream>
#include <vector>
#include "acl/acl.h"
// #include "aclnnop/aclnn_grouped_matmul_v4.h"
#include <fstream>
#include "aclnn_replace_token.h"
#include "acl/acl_op_compiler.h"
#include <cassert>
#include <limits>
#include <string>
#include <cstdint>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)
  template <typename T>
  bool ReadIntegersFromBinary(const std::string& file_path, std::vector<T>& data) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening binary file: " << file_path << std::endl;
        return false;
    }
  
    // Get the file size
    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
  
    // Verify that the file size is an integer multiple
    const size_t num_elements = static_cast<size_t>(file_size / sizeof(T));
    if (file_size % sizeof(T) != 0) {
        std::cerr << "Invalid binary file format" << std::endl;
        return false;
    }
  
    // Preallocate memory
    data.resize(num_elements);
  
    // Reading Data
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    if (!file.good()) {
        std::cerr << "Failed to read binary data" << std::endl;
        return false;
    }
    return true;
  }
  
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Fixed writing, AscendCL initialization
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor_New(const std::vector<int64_t>& shape, void** deviceAddr,
                        aclDataType dataType, aclTensor** tensor,const std::string& binFilePath = "") {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  std::vector<T> hostData(GetShapeSize(shape));
  // for (size_t i = 0; i < hostData.size(); ++i) {
  //   hostData[i] = i+1;
  // }
  // hostData[255] = 415;
  if(!binFilePath.empty()){
    ReadIntegersFromBinary<T>(binFilePath,hostData);  
  }else{
    std::fill(hostData.begin(), hostData.end(), T(1));
  }        
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}
template <typename T>
int CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor, aclFormat format = aclFormat::ACL_FORMAT_ND, T data = 1,const std::string& binFilePath = "") {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to apply for device-side memory
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy the host-side data to the device-side memory
  std::vector<T> hostData(GetShapeSize(shape));
  if(!binFilePath.empty()){
    ReadIntegersFromBinary<T>(binFilePath,hostData);  
  }else{
    std::fill(hostData.begin(), hostData.end(), T(1));
  }           
  // std::cout<<GetShapeSize(shape)<<std::endl;
  // std::cout<<static_cast<int>(hostData[0])<<std::endl;
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of continuous tensors
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  // if(shape.size() > 2){
  //   strides[1] = 1;
  //   strides[2] = 7168;
  // }
  // Call the aclCreateTensor interface to create aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}


int CreateAclTensorList(const std::vector<std::vector<int64_t>>& shapes, void** deviceAddr,
                        aclDataType dataType, aclTensorList** tensor, aclFormat format = aclFormat::ACL_FORMAT_ND,const std::string& binFilePath = "") {
  int size = shapes.size();
  aclTensor* tensors[size];
  for (int i = 0; i < size; i++) {
    int ret;
    if (dataType == aclDataType::ACL_INT8) {
      int8_t data = 1;
      ret = CreateAclTensor<int8_t>(shapes[i], deviceAddr + i, dataType, tensors + i, format, data,binFilePath);
    } else if (dataType == aclDataType::ACL_BF16) {
      uint16_t data = 0x3f80;
      ret = CreateAclTensor<uint16_t>(shapes[i], deviceAddr + i, dataType, tensors + i, format, data,binFilePath);
    } else if (dataType == aclDataType::ACL_FLOAT) {
      float data = 1.0f;
      ret = CreateAclTensor<float>(shapes[i], deviceAddr + i, dataType, tensors + i, format, data,binFilePath);
    } else if (dataType == aclDataType::ACL_INT64) {
      int64_t data = 1;
      ret = CreateAclTensor<float>(shapes[i], deviceAddr + i, dataType, tensors + i, format, data,binFilePath);
    } else if (dataType == aclDataType::ACL_FLOAT16) {
      uint16_t data = 0x3f80;
      ret = CreateAclTensor<float>(shapes[i], deviceAddr + i, dataType, tensors + i, format, data,binFilePath);
    }
    else {
      LOG_PRINT("error!");
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }
  *tensor = aclCreateTensorList(tensors, size);
  return ACL_SUCCESS;
}


int main(int argc, char **argv) {
  // 1. (Fixed writing) device/stream initialization, refer to the AscendCL external interface list
  // Fill in deviceId according to your actual device
  int32_t deviceId = 8;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // checkProcess according to your needs
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct input and output, which needs to be customized according to the API interface
  //shape2 :m=416 n=7168 k =128
  
  std::string input_forked_token_ids_path = "";
  std::string input_last_step_out_put_token_ids_path = "";
  int64_t sequenceLength=5;
  if(argc<3){
    input_forked_token_ids_path = "./input/forked_token_ids.bin";
    input_last_step_out_put_token_ids_path = "./input/last_step_out_put_token_ids.bin";
    std::cout<<"use default input path"<<std::endl;
  }else{
    sequenceLength = std::stoi(argv[1]);
    input_forked_token_ids_path = argv[2];
    input_last_step_out_put_token_ids_path = argv[3];
    std::cout<<"use custom input path"<<std::endl;
  }
  
  std::cout<<input_forked_token_ids_path<<std::endl;
  std::cout<<input_last_step_out_put_token_ids_path<<std::endl;
  // "416,7168;256,7168,256;;256,256;;;;256;416"
  std::vector<std::vector<int64_t>> forkedTokenIdsShape = {{sequenceLength}};
  std::vector<std::vector<int64_t>> lastStepOutPutTokenIdsShape= {{sequenceLength}};
  std::vector<std::vector<int64_t>> outShape = {{sequenceLength}};
  void* forkedTokenIdsDeviceAddr[1];
  void* lastStepOutPutTokenIdsDeviceAddr[1];
  void* outDeviceAddr[1];
  aclTensor* forkedTokenIds = nullptr;
  aclTensor* lastStepOutPutTokenIds = nullptr;
  aclTensor* out = nullptr;
  aclFormat format = aclFormat::ACL_FORMAT_ND;

  // Create x aclTensor
  ret = CreateAclTensor<int32_t>(forkedTokenIdsShape[0], forkedTokenIdsDeviceAddr, aclDataType::ACL_INT32, &forkedTokenIds, format, 1,input_forked_token_ids_path);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create weight aclTensor
  ret = CreateAclTensor<int64_t>(lastStepOutPutTokenIdsShape[0], lastStepOutPutTokenIdsDeviceAddr, aclDataType::ACL_INT64, &lastStepOutPutTokenIds, format,1, input_last_step_out_put_token_ids_path);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // ret = CreateAclTensor<int32_t>(outShape[0], outDeviceAddr, aclDataType::ACL_INT32, &out, format,1);
  // CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 3. Call the CANN operator library API

  ret = aclnnReplaceTokenGetWorkspaceSize(forkedTokenIds, lastStepOutPutTokenIds, forkedTokenIds, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReplaceTokenGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Apply for device memory based on the workspaceSize calculated by the first section of the interface
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second section of aclnnGroupedMatmulV4 interface
  ret = aclnnReplaceToken(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReplaceToken failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Synchronously wait for task execution to complete
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Get the output value and copy the result in the device memory to the host side. This needs to be modified according to the specific API interface definition.
  
  auto size = GetShapeSize(forkedTokenIdsShape[0]);
  std::vector<int32_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), forkedTokenIdsDeviceAddr[0],
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("Copy output[%zu] success", 0);
  std::string filePath ="./output/output_out.bin";
  int fd =
      open(filePath.c_str(), 02 |  0100 | 01000, 0400 | 0200);
  if (fd < 0) {
    ERROR_LOG("Open file failed. path = %s", filePath.c_str());
  }

  auto writeSize = write(fd, resultData.data(), size*sizeof(resultData[0]));
  close(fd);
  if (writeSize != size*sizeof(resultData[0])) {
    
    ERROR_LOG("Write file Failed.");
  }
  // close(fd);
  LOG_PRINT("Write file success.");
  // WriteFile("./output/output_z.bin",resultData);


  // 6. Release aclTensor and aclScalar, which need to be modified according to the interface definition of the specific API
  aclDestroyTensor(forkedTokenIds);
  aclDestroyTensor(lastStepOutPutTokenIds);

  // 7. Release device resources, which needs to be modified according to the interface definition of the specific API
  for (int i = 0; i < 1; i++) {
    aclrtFree(forkedTokenIdsDeviceAddr[i]);
    aclrtFree(lastStepOutPutTokenIdsDeviceAddr[i]);
  }
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}