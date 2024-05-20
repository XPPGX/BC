#!/bin/bash

folder_path="../dataset/"

# datasets=("soc-flickr.mtx" "loc-Gowalla.mtx" "tech-RL-caida.mtx" "soc-youtube.mtx" "web-NotreDame-OK.mtx" "youtube.txt")
# datasets=("amazon0302-OK.mtx" "web-sk-2005.mtx" "web-NotreDame-OK.mtx" "soc-youtube.mtx" "web-Google-Ok2.mtx" "youtube.txt" "road-roadNet-CA.mtx")
datasets=("wikiTalk.mtx" "web-wikipedia2009.mtx")
# datasets=()
# while IFS= read -r -d $'\0' file; do
#     filename=$(basename "$file")
#     datasets+=("$filename")
# done < <(find "$folder_path" -maxdepth 1 -type f -print0)

# echo "Files in the folder:"
# for dataset in "${datasets[@]}"; do
#     echo "${dataset%.*}"
# done

make clean && make

# 循环执行五个不同的数据集实验
for dataset in "${datasets[@]}"; do
    # 运行实验

    nohup ./a "../dataset/$dataset" &> "D1_AP_CC_${dataset%.*}_Time208.txt" &
    echo "Running experiment with $dataset"
    # 等待当前实验完成
    wait $!
    echo "Experiment with $dataset completed."
done

# echo "All experiments completed."
