#!/bin/zsh

data=$1
data_upper=${data:u}

echo $data
echo $data_upper

echo "Individual training"
python src/train/individual-train.py  --outdir 'experiments' --model 'mlp' --seed 1234 --k 8 --dataset 'src.data.data.GroupData'$data_upper
echo "Group training"
python src/train/group-train-all.py  --outdir 'experiments' --model 'experiments/'$data'/mlp_k8_ds'$data'_seed1234.h5' --seed 1234 --k 8 --dataset 'src.data.data.GroupData'$data_upper
echo "Evaluation"
python src/eval/eval.py  --outdir 'results' --modelFile 'experiments/'$data'/mlp_k8_ds'$data'_seed1234.h5' --seed 1234 --k 8 --dataset 'src.data.data.GroupData'$data_upper

