#screen -dmS clu0 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='0';./exp_clustering.sh 2";
#screen -dmS clu1 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='1';./exp_clustering.sh 1";
screen -dmS A bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='0';python exp_clustering.py 0 1 standard";
screen -dmS B bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='1';python exp_clustering.py 0 1 universal";
screen -dmS C bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='2';python exp_clustering.py 0 0 standard";
screen -dmS D bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='3';python exp_clustering.py 0 0 universal";
screen -dmS E bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='4';python exp_clustering.py 1 1 standard";
screen -dmS F bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='5';python exp_clustering.py 1 1 universal";
screen -dmS G bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='6';python exp_clustering.py 1 0 standard";
screen -dmS H bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='7';python exp_clustering.py 1 0 universal";




