#screen -dmS clu0 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='0';./exp_clustering.sh 2";
#screen -dmS clu1 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='1';./exp_clustering.sh 1";
screen -dmS clu2 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='0';python exp_fa.py 2";



