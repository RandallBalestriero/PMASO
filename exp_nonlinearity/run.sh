screen -dmS nonlinearityMnistLocal bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='0';./exp_nonlinearity.sh MNIST local";
screen -dmS nonlinearityFlippedMnistLocal bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='1';./exp_nonlinearity.sh flippedMNIST local";

screen -dmS nonlinearityMnistGlobal bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='6';./exp_nonlinearity.sh MNIST global";
screen -dmS nonlinearityFlippedMnistGlobal bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='7';./exp_nonlinearity.sh flippedMNIST global";




