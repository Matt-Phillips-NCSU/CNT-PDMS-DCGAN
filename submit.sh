#!/bin/bash
#BSUB -W 72:00
#BSUB -n 1
#BSUB -o out.%J
#BSUB -e err.%J
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -R "select[!gtx1080]"
#BSUB -R "select[!rtx2080]"
#BSUB -R "select[!p100]"
#BSUB -R "rusage[mem=400GB]"
source ~/.bashrc
conda activate /usr/local/usrapps/zikry/mphilli2/mlflow_env
python DCGAN_320Pix_4Layer_FinalTrainAndTest.py
conda deactivate
