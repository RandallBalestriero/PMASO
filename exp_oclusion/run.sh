#screen -dmS oclusionMLPixel1 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='2';./exp_oclusion.sh MLP pixel 0.1";
#screen -dmS oclusionextraCNNPixel1 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='6';./exp_oclusion.sh MLP pixel 0.1";

#screen -dmS A bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='2';./exp_oclusion.sh 9 MLP pixel 0.05";
screen -dmS A bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='3';./exp_oclusion.sh 8 MLP box 20";
screen -dmS A bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='4';./exp_oclusion.sh 9 MLP box 20";


#screen -dmS oclusionMLPBox20 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='4';./exp_oclusion.sh MLP box 20";



