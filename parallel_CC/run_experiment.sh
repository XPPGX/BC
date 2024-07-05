#!/bin/bash

folder_path="../dataset/"

# datasets=("soc-flickr.mtx" "loc-Gowalla.mtx" "tech-RL-caida.mtx" "soc-youtube.mtx" "web-NotreDame-OK.mtx" "youtube.txt")
# datasets=("amazon0302-OK.mtx" "web-sk-2005.mtx" "web-NotreDame-OK.mtx" "soc-youtube.mtx" "web-Google-Ok2.mtx" "youtube.txt" "road-roadNet-CA.mtx")
# datasets=("wikiTalk.mtx" "web-wikipedia2009.mtx")
# datasets=()
# while IFS= read -r -d $'\0' file; do
#     filename=$(basename "$file")
#     datasets+=("$filename")
# done < <(find "$folder_path" -maxdepth 1 -type f -print0)

# echo "Files in the folder:"
# for dataset in "${datasets[@]}"; do
#     echo "${dataset%.*}"
# done

# datasets=("musae_git.txt")
# datasets=("ia-email-EU.mtx")
# datasets=("tech-WHOIS.mtx" "ca-HepPh.mtx" "caida.mtx")
# datasets=("karate.txt" "dolphins.txt" "football.txt" "polbooks.txt" "ia-fb-messages.mtx" "tech-WHOIS.mtx" "ca-HepPh.mtx" "caida.mtx" "ia-email-EU.mtx" "musae_git.txt")

datasets=("web-sk-2005.mtx" "Slashdot0811-OK.mtx" "musae_git.txt" "loc-Gowalla.mtx" "tech-RL-caida.mtx" "dblp.txt" "amazon.txt" "soc-flickr.mtx" "twitch_gamers.txt" "amazon0302-OK.mtx" "soc-youtube.mtx" "youtube.txt" "web-Google-Ok2.mtx" "wikiTalk.mtx" "web-wikipedia2009.mtx")

make clean && make

# 循环执行五个不同的数据集实验
# for dataset in "${datasets[@]}"; do
#     # 运行实验
#     # echo -e "iteration $i\n" >> "ShareBased_Time208.txt"

#     nohup ./a "../dataset/$dataset" &> "D1_AP_CC_${dataset%.*}_Time208.txt" &
#     echo "Running experiment with $dataset"
#     # 等待当前实验完成
#     wait $!
#     echo "Experiment with $dataset completed."
# done
# echo -e "\n" >> "ShareBased_Time208.txt"
# echo "All experiments completed."

# 小資料集的實驗script

for i in {1..8}; do
    echo -e "iteration $i\n" >> "parallel_ShareBased_Time208.txt"
    # 循环执行五个不同的数据集实验
    for dataset in "${datasets[@]}"; do
        # 运行实验

        nohup ./a "../dataset/$dataset" &> "para_D1_AP_CC_${dataset%.*}_Time208.txt" &
        echo "$(date +"%Y-%m-%d %H:%M:%S") [Running]\texperiment with $dataset"
        # 等待当前实验完成
        wait $!
        echo "$(date +"%Y-%m-%d %H:%M:%S") [Finished]\tExperiment with $dataset completed."
    done
done
echo -e "\n" >> "parallel_ShareBased_Time208.txt"