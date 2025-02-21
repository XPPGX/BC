#!/bin/bash

folder_path="../dataset/"
datasets=("ia-email-EU.mtx" "musae_git.txt")
# datasets=("tech-WHOIS.mtx" "ca-HepPh.mtx" "caida.mtx")
# datasets=("karate.txt" "dolphins.txt" "football.txt" "polbooks.txt" "ia-fb-messages.mtx")

make clean && make

for i in {1..1}; do
    echo -e "iteration $i\n" >> "Bellman_Ford_Time208.txt"
    # 循环执行五个不同的数据集实验
    for dataset in "${datasets[@]}"; do
        # 运行实验

        nohup ./a "../dataset/$dataset" &> "D1_AP_CC_${dataset%.*}_Time208.txt" &
        echo "[Running]\texperiment with $dataset"
        # 等待当前实验完成
        wait $!
        echo "[Finished]\tExperiment with $dataset completed."
    done
done
echo -e "\n" >> "Bellman_Ford_Time208.txt"
# echo "All experiments completed."
