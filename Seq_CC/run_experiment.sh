#!/bin/bash

folder_path="../dataset/"

datasets=()
while IFS= read -r -d $'\0' file; do
    filename=$(basename "$file")
    datasets+=("$filename")
done < <(find "$folder_path" -maxdepth 1 -type f -print0)

echo "Files in the folder:"
for dataset in "${datasets[@]}"; do
    echo "$dataset"
done




# 循环执行五个不同的数据集实验
for dataset in "${datasets[@]}"; do
    # 运行实验
    nohup ./a "../dataset/$dataset" &> "CC_${dataset%}_ans.txt" &
    echo "Running experiment with $dataset"
    # 等待当前实验完成
    wait $!
    echo "Experiment with $dataset completed."
done

echo "All experiments completed."
