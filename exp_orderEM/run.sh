screen -dmS orderEM00 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='0';./exp_orderEM.sh 0 0";
#screen -dmS orderEM01 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='1';./exp_orderEM.sh 0 1";
#screen -dmS orderEM10 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='6';./exp_orderEM.sh 1 0";
screen -dmS orderEM11 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='7';./exp_orderEM.sh 1 1";





