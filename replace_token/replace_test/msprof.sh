# export  ASCEND_RT_VISIBLE_DEVICES=0f 
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
m=$1
n=$2
k=$3
cd $CURRENT_DIR
cd ../../group_gemm/
# bash build.sh
unset ASCEND_CUSTOM_OPP_PATH
./build_out/custom_opp_openEuler_x86_64.run
cd $CURRENT_DIR
chown -R root:root .
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascend910B2C/lib:$LD_LIBRARY_PATH 
# cd ./build
# make
# cd ..
msprof op simulator --output=./output_op_simulator/0608_multicube_index_m${m}_n${n}_k${k} ./build/bin/test_grouped_matmul ${m} ${n} ${k}
# msprof op --warm-up=5 --output=./output/0606_multicube_index_op_group_m${m}_n${n}_k${k}   --aic-metrics=Default,Roofline,Occupancy ./build/bin/test_grouped_matmul ${m} ${n} ${k}
# msprof  --output=./output/system/0608_multicube_index_group_m${m}_n${n}_k${k}  ./build/bin/test_grouped_matmul ${m} ${n} ${k}
# --replay-mode=application

# mssanitizer
# mssanitizer --tool=racecheck -t racecheck -t initcheck ./build/bin/test_grouped_matmul ${m} ${n} ${k}
# npu-smi info -t power -i 0 -c 0
# chown -R limenxin1:limenxin1 . 
