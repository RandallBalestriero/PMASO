screen -dmS oclusionCNNPixel05 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='2';./exp_oclusion.sh CNN pixel 0.05";
screen -dmS oclusionCNNPixel1 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='3';./exp_oclusion.sh CNN pixel 0.1";

screen -dmS oclusionBox5 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='4';./exp_oclusion.sh CNN box 5";
screen -dmS oclusionBox10 bash -c "sleep 2;export CUDA_VISIBLE_DEVICES='5';./exp_oclusion.sh CNN box 10";



