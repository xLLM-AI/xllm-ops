for i in {1..10}
do
  bash run.sh > log_$i.txt 2>&1
done