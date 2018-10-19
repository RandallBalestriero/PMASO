screen -dmS FA8 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='2';./exp_fa.sh 8";
screen -dmS FA16 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='0';./exp_fa.sh 16";
screen -dmS FA32 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='1';./exp_fa.sh 32";



