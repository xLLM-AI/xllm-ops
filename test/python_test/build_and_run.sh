#!/bin/bash

BASE_DIR=$(pwd)

# Remove previous build artifacts
rm -rf ${BASE_DIR}/build
rm -rf ${BASE_DIR}/dist
rm -rf ${BASE_DIR}/*.egg-info

# Build and install the package
python3 setup.py build bdist_wheel
cd ${BASE_DIR}/dist
pip3 install --force-reinstall *.whl

# Run tests
cd ${BASE_DIR}/test
python3 select_unshared_kv.py
if [ $? -ne 0 ]; then
    echo "[ERROR]: Run select_unshared_kv test failed!"
fi
echo "[INFO]: Run select_unshared_kv test success!"

python3 cache_unshared_kv.py
if [ $? -ne 0 ]; then
    echo "[ERROR]: Run cache_unshared_kv test failed!"
fi
echo "[INFO]: Run cache_unshared_kv test success!"

# export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascend910B2C/lib:$LD_LIBRARY_PATH
# msprof op simulator python3 cache_unshared_kv.py
