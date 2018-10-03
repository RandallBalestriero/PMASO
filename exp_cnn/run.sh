screen -dmS CIFAR0 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='2';./exp_cnn.sh CIFAR 0";
screen -dmS CIFAR1 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='2';./exp_cnn.sh CIFAR 1";
screen -dmS CIFAR2 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='2';./exp_cnn.sh CIFAR 2";
screen -dmS CIFAR9 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='2';./exp_cnn.sh CIFAR 9";



