# Deep Neural Aggregation for Recommending Items to Group of Users 

This repository contains the source code of the experiments run for a paper titled "*Deep Neural Aggregation for Recommending Items to a Group of Users*". This work has been submitted and is currently under review. You can read its preprint on arXiv.: [https://arxiv.org/abs/2307.09447](https://arxiv.org/abs/2307.09447).

## python RS data library

This project use a library that manage all preprocessing for recommender system datasets. It has not been plubished as a python package yet, so it is included in this repository. You can added it to your path:

```
PYTHONPATH="/workspace/rs-data-python:."
export PYTHONPATH
```

### Datasets

All experiments use the datasets from the java library [CF4J](https://cf4j.etsisi.upm.es/)

## Project Layout

This is the project layout with an explanation about what you can find in each directory.

```txt
- data
  \- data for experiments
- experiments
  \- artefacts produced by execution of experiments
  \- models h5, data files, etc
- results
  \- results generated after evaluation
- notebooks
  \- pre and post data processing
  \- presentation of data
- src
  \- data 
  \- eval
  \- models
  \- train
  \- utils
```

## Execution of experiments

All scripts and code has a seed initialization, you can reproduce this experiment results doing the folling tasks:

1.- Clone the repository
2.- Setup the python path for datasets
3.- Download the datasets
4.- Generate the synthetyc groups
5.- Split groups in train/val/test
6.- Train the individual models
7.- Train the group models
8.- Eval the trained models

When you have generated the results you can generate the graphics with the scripts in the ```notebooks``` folder. Papers final graphs are in notebook ```notebooks/results-paper```.

### Group generation

In the function ```generate_group``` the first argument is mandatory and is the code of the dataset. The second is optional and indicates the group size (Usefull for big datasets like Anime).

```python
python -c "from src.data.data import generate_groups; generate_groups('ml100k')"
```

It is important to set the random seed in order to reproduce the experiments. All group data has been generated with the default random_seed (value 37 see: rs-data-python/data_utlis.py).

```python
python -c "
from data_utils import init_random;
init_random()
from src.data.data import generate_groups
generate_groups('anime',2)"
```

#### Split Test-Train

```
zsh src/data/groups-spliter.sh data/grupos/ml1m
```

### Train MLP with individual data

```
python src/train/individual-train.py  --outdir 'experiments' --model mlp --seed 1234 --k 8 --dataset 'src.data.data.GroupDataFT'
python src/train/individual-train.py  --outdir 'experiments' --model mlp --seed 1234 --k 8 --dataset 'src.data.data.GroupDataML1M'
python src/train/individual-train.py  --outdir 'experiments' --model mlp --seed 1234 --k 8 --dataset 'src.data.data.GroupDataANIME'
python src/train/individual-train.py  --outdir 'experiments' --model mlp --seed 1234 --k 8 --dataset 'src.data.data.GroupDataML100K'
python src/train/individual-train.py  --outdir 'experiments' --model gmf --seed 1234 --k 8 --dataset 'src.data.data.GroupDataFT'
python src/train/individual-train.py  --outdir 'experiments' --model gmf --seed 1234 --k 8 --dataset 'src.data.data.GroupDataML1M'
python src/train/individual-train.py  --outdir 'experiments' --model gmf --seed 1234 --k 8 --dataset 'src.data.data.GroupDataANIME'
python src/train/individual-train.py  --outdir 'experiments' --model gmf --seed 1234 --k 8 --dataset 'src.data.data.GroupDataML100K'
```

### Train Aggregator as MLP

```
python src/train/group-train-all.py  --outdir 'experiments' --model 'experiments/ft/mlp_k8_dsft_seed1234.h5' --seed 1234 --k 8 --dataset 'src.data.data.GroupDataFT'
python src/train/group-train-all.py  --outdir 'experiments' --model 'experiments/ml100k/mlp_k8_dsml100k_seed1234.h5' --seed 1234 --k 8 --dataset 'src.data.data.GroupDataML100K'
python src/train/group-train-all.py  --outdir 'experiments' --model 'experiments/ml1m/mlp_k8_dsml1m_seed1234.h5' --seed 1234 --k 8 --dataset 'src.data.data.GroupDataML1M'
python src/train/group-train-all.py  --outdir 'experiments' --model 'experiments/anime/mlp_k8_dsanime_seed1234.h5' --seed 1234 --k 8 --dataset 'src.data.data.GroupDataANIME'
```

Train with bash variables
```
data="ml1m";model="mlp"
data=${data:l}
data_upper=${data:u}
python src/train/group-train-all.py  --outdir 'experiments' --model 'experiments/'$data'/'$model'_k8_ds'$data'_seed1234.h5' --seed 1234 --k 8 --dataset 'src.data.data.GroupData'$data_upper
```

Train with bash variables including group size
```
data="ml1m";model="mlp";group_size=2
data=${data:l}
data_upper=${data:u}
python src/train/group-train-all.py  --outdir 'experiments' --model 'experiments/'$data'/'$model'_k8_ds'$data'_seed1234.h5' --seed 1234 --k 8 --dataset 'src.data.data.GroupData'$data_upper --group_size $group_size
```

Train with bash variables, group size, agg function
```
data="anime";model="gmf";group_size=2;agg="mode"
data=${data:l}
data_upper=${data:u}
python src/train/group-train.py  --outdir 'experiments' --model 'experiments/'$data'/'$model'_k8_ds'$data'_seed1234.h5' --seed 1234 --k 8 --dataset 'src.data.data.GroupData'$data_upper --group_size $group_size --agg $agg
```


### Eval

```
python src/eval/eval.py  --outdir 'results' --modelFile 'experiments/ft/mlp_k8_dsft_seed1234.h5' --modelName mlp --seed 1234 --k 8 --dataset 'src.data.data.GroupDataFT'
python src/eval/eval.py  --outdir 'results' --modelFile 'experiments/ml1m/mlp_k8_dsml1m_seed1234.h5' --modelName mlp --seed 1234 --k 8 --dataset 'src.data.data.GroupDataML1M'
python src/eval/eval.py  --outdir 'results' --modelFile 'experiments/anime/mlp_k8_dsanime_seed1234.h5' --modelName mlp --seed 1234 --k 8 --dataset 'src.data.data.GroupDataANIME'
```

Eval with bash variables
```
data="ml1m";model="mlp"
data=${data:l}
data_upper=${data:u}
python src/eval/eval.py  --outdir 'results' --modelFile 'experiments/'$data'/'$model'_k8_ds'$data'_seed1234.h5' --modelName $model --seed 1234 --k 8 --dataset 'src.data.data.GroupData'$data_upper
```

### Execution per dataset

```zsh
zsh src/runner.sh ml1m mlp|gmf
zsh src/runner.sh ft mlp|gmf
zsh src/runner.sh anime mlp|gmf
```


## Previous results and discarded ideas

**Discarded** Generate a multihot vector representing the group's users and feed it to the individual model. Better performance by generating the group in the latent space (connecting it to the dense layer of the individual model). Group representation as classic Multihot vector works in some datasets, like FT with low number of users but no work at all for Anime. ![MINMAX](discarded/agg-as-dense.png)

**Discarded** Min-Max as 'Y' to train the group aggregation get worse results. ![MINMAX](discarded/min-max.png)

**Note** Group representation in latent factor space can have negative values. Last layer in group MLP must be linear


## Useful dev info

### Tareas

(x) Aumentar tamaño de grupos. Mantener grupos generados de test.
(x) Pegar salida a embedding del modelo individual
(x) Generar Test-Train y usar validation
(x) Poner early stop y aumentar el número de EPOCHs
(x) Probar entrenar: MAX, MIN, MEAN, MEDIAN, MODA
(x) Sacar info de GMF, NCF con variaciones
(x) Pintar STD
(x) Meter MLP y GMF con softmax
(x) ML1M
(x) MLP-Evaluado y entrenado
(x) Entrenando GMF
(x) Entrenando a Anime de nuevo con MLP y GMF para individuos.
(x) Elegir mejor GMF y NCF

### Gráficas

- Ejecución de todas las agregaciones
- Hacer column-boxplot teniendo la misma escala.

Figura
Para cada modelo (GMF-MLP)
Para cada función de agregación (min, max, mean, median, mode)
Para cada tamaño de grupo (2-10)
Dos métricas MAE y MSE

#### GMF
https://stackoverflow.com/questions/45875143/seaborn-making-barplot-by-group-with-asymmetrical-custom-error-bars
https://stackoverflow.com/questions/35978727/how-add-asymmetric-errorbars-to-pandas-grouped-barplot
https://stackoverflow.com/questions/23000418/adding-error-bars-to-grouped-bar-plot-in-pandas
