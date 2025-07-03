CURRENT_DIR=$(pwd)
cd $CURRENT_DIR
cd ../replace_token
bash build.sh
unset ASCEND_CUSTOM_OPP_PATH
./build_out/custom_opp_openEuler_x86_64.run
cd $CURRENT_DIR
bash build.sh
forked_token_ids_path="${CURRENT_DIR}/input/forked_token_ids.bin"
last_step_out_put_token_ids_path="${CURRENT_DIR}/input/last_step_out_put_token_ids.bin"
output_out_file_path="${CURRENT_DIR}/input/output_out.bin"
acl_output_out_file_path="${CURRENT_DIR}/output/output_out.bin"
log_file_path="${CURRENT_DIR}/log"
for m in 4 8 16 32 64 128 256 512 1024
do  
    echo "m: $m start process"
    python3 scripts/gen_data.py $m
    ./build/bin/test_replace_token $m $forked_token_ids_path $last_step_out_put_token_ids_path
    md5sum $output_out_file_path $acl_output_out_file_path >> ${log_file_path}/log_${m}.txt
    # python3 scripts/read_file.py
done

# mssanitizer --tool=memcheck -t racecheck -t initcheck ./build/bin/test_replace_token 5 $forked_token_ids_path $last_step_out_put_token_ids_path
# bash scripts/verify.sh
# ./build/bin/test_replace_token 5 $forked_token_ids_path $last_step_out_put_token_ids_path
# python3 scripts/read_file.py