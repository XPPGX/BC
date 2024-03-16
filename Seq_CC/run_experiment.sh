#!/bin/bash
#my dataset
datasets=("karate.txt" "dblp.txt" "amazon.txt" "youtube.txt")
#cost too much time
# datasets=("road-roadNet-CA.mtx")

#senior dataset
# datasets=("loc-Gowalla.mtx" "Slashdot0811-OK.mtx" "soc-flickr.mtx" "web-BerkStan-OK.mtx" "web-sk-2005.mtx" "web-Stanford.mtx")

# 循环执行五个不同的数据集实验
for dataset in "${datasets[@]}"; do
    # 运行实验
    nohup ./a "../dataset/$dataset" &> "CC_${dataset%.txt}_ans.txt" &
    echo "Running experiment with $dataset"
    # 等待当前实验完成
    wait $!
    echo "Experiment with $dataset completed."
done

echo "All experiments completed."