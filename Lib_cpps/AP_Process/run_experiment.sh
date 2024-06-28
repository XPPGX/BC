#!/bin/bash
#my dataset
# datasets=("karate.txt" "dblp.txt" "amazon.txt" "youtube.txt" "Slashdot0811-OK.mtx" "loc-Gowalla.mtx" "soc-flickr.mtx" "musae_git.txt" "tech-RL-caida.mtx" "twitch_gamers.txt")
# datasets=("karate.txt" "Slashdot0811-OK.mtx" "loc-Gowalla.mtx" "musae_git.txt" "tech-RL-caida.mtx")
# datasets=("web-NotreDame-OK.mtx" "soc-youtube.mtx")
#cost too much time
# datasets=("road-roadNet-CA.mtx")

#senior dataset
# datasets=("loc-Gowalla.mtx" "Slashdot0811-OK.mtx" "soc-flickr.mtx" "web-BerkStan-OK.mtx" "web-sk-2005.mtx" "web-Stanford.mtx")

datasets=("LiveJournal.txt")
# datasets=("musae_git.txt" "tech-RL-caida.mtx" "twitch_gamers.txt" "web-wikipedia2009.mtx" "wikiTalk.mtx" "web-Google-Ok2.mtx" "soc-Epinions1-Ok2.mtx")



# 循环执行五个不同的数据集实验
for dataset in "${datasets[@]}"; do
    # 运行实验
    nohup ./a "../../dataset/$dataset" &> "AP_process_${dataset%.txt}.txt" &
    echo "Running experiment with $dataset"
    # 等待当前实验完成
    wait $!
    echo "Experiment with $dataset completed."
done

echo "All experiments completed."