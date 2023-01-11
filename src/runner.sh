#!/bin/zsh

data=$1
data=${data:l}
data_upper=${data:u}

model=$2

echo "Dataset: $data - $data_upper"
echo "Model $model"

echo "Individual training"
python src/train/individual-train.py  --outdir 'experiments' --model $model --seed 1234 --k 8 --dataset 'src.data.data.GroupData'$data_upper
echo "Group training"
python src/train/group-train-all.py  --outdir 'experiments' --model 'experiments/'$data'/'$model'_k8_ds'$data'_seed1234.h5' --seed 1234 --k 8 --dataset 'src.data.data.GroupData'$data_upper
echo "Evaluation"
python src/eval/eval.py  --outdir 'results' --modelFile 'experiments/'$data'/'$model'_k8_ds'$data'_seed1234.h5' --modelName $model --seed 1234 --k 8 --dataset 'src.data.data.GroupData'$data_upper

