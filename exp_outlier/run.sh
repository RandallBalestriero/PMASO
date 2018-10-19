#screen -dmS outlier1 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='0';./exp_outlier.sh MNIST 0"; # 0 1 2 3 7 8 9 done
screen -dmS outlier2 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='1';./exp_outlier.sh CIFAR 4";
screen -dmS outlier1 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='0';./exp_outlier.sh CIFAR 5";
screen -dmS outlier3 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='2';./exp_outlier.sh CIFAR 6";




#screen -dmS FA16 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='0';./exp_fa.sh 16";
#screen -dmS FA32 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='1';./exp_fa.sh 32";



