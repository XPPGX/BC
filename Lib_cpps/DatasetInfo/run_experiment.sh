#!/bin/bash
#my dataset
# datasets=("karate.txt" "dblp.txt" "amazon.txt" "youtube.txt" "road-roadNet-CA.mtx" "loc-Gowalla.mtx" "Slashdot0811-OK.mtx" "soc-flickr.mtx" "web-BerkStan-OK.mtx" "web-sk-2005.mtx" "web-Stanford.mtx" "musae_git.txt" "tech-RL-caida.mtx" "twitch_gamers.txt" "web-Google-Ok2.mtx" "wikiTalk.mtx")
# datasets=("musae_git.txt" "tech-RL-caida.mtx" "twitch_gamers.txt" "web-Google-Ok2.mtx" "wikiTalk.mtx")
datasets=("LiveJournal.txt")
# 循环执行五个不同的数据集实验
for dataset in "${datasets[@]}"; do
    # 运行实验
    nohup ./a "../../dataset/$dataset" &> "Degree_distribution_${dataset%.txt}.txt" &
    echo "Running experiment with $dataset"
    # 等待当前实验完成
    wait $!
    echo "Experiment with $dataset completed."
done

echo "All experiments completed."