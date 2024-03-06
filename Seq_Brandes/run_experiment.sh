#!/bin/bash
datasets=("karate.txt" "dblp.txt" "amazon.txt" "youtube.txt" "LiveJournal.txt")

# 循环执行五个不同的数据集实验
for dataset in "${datasets[@]}"; do
    # 运行实验
    nohup ./a "../dataset/$dataset" &> "CC_shareBased_${dataset%.txt}.txt" &
    echo "Running experiment with $dataset"
    # 等待当前实验完成
    wait $!
    echo "Experiment with $dataset completed."
done

echo "All experiments completed."