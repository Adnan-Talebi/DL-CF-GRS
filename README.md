# Deep Neural Aggregation for Recommending Items to Group of Users 

This repository contains the source code of the experiments run for a paper titled "*Deep Neural Aggregation for Recommending Items to a Group of Users*". This work has been submitted to the Information Sciences journal (ref. INS-D-23-4969) and is currently under review. You can read its preprint on arXiv.: [https://arxiv.org/abs/2307.09447](https://arxiv.org/abs/2307.09447).

## Data library

This project use an external library.

```
PYTHONPATH="/workspace/rs-data-python:."
export PYTHONPATH
```

## Project Layout

```txt
- data
  \- data for experiments
- experiments
  \- artefacts produced by execution of experiments
  \- models, data files, etc.
- results
  \- infor generated after evaluation
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

## Execution example

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
python src/train/group-train-all.py  --outdir 'experiments' --model 'experiments/ml1m/mlp_k8_dsml1m_seed1234.h5' --seed 1234 --k 8 --dataset 'src.data.data.GroupDataML100K'
python src/train/group-train-all.py  --outdir 'experiments' --model 'experiments/ml1m/mlp_k8_dsml1m_seed1234.h5' --seed 1234 --k 8 --dataset 'src.data.data.GroupDataML1M'
python src/train/group-train-all.py  --outdir 'experiments' --model 'experiments/anime/mlp_k8_dsanime_seed1234.h5' --seed 1234 --k 8 --dataset 'src.data.data.GroupDataANIME'
```

Train generic
```
data="ml1m";model="mlp"
data=${data:l}
data_upper=${data:u}
python src/train/group-train-all.py  --outdir 'experiments' --model 'experiments/'$data'/'$model'_k8_ds'$data'_seed1234.h5' --seed 1234 --k 8 --dataset 'src.data.data.GroupData'$data_upper
```

Train generic group size
```
data="ml1m";model="mlp";group_size=2
data=${data:l}
data_upper=${data:u}
python src/train/group-train-all.py  --outdir 'experiments' --model 'experiments/'$data'/'$model'_k8_ds'$data'_seed1234.h5' --seed 1234 --k 8 --dataset 'src.data.data.GroupData'$data_upper --group_size $group_size
```

Train specific group size, agg function
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

Only Eval
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


## Previous results

**Discarded** Generate a multihot vector representing the group's users and feed it to the individual model. Better performance by generating the group in the latent space (connecting it to the dense layer of the individual model). Group representation as classic Multihot vector works in some datasets, like FT with low number of users but no work at all for Anime. ![MINMAX](discarded/agg-as-dense.png)

**Discarded** Min-Max as 'Y' to train the group aggregation get worse results. ![MINMAX](discarded/min-max.png)

**Note** Estoy ejecutando la evaluación de los modelos sencillos. Igual que el otro paper para ver qué hacen los grupos en anime.
Con bobi Anime tenía más error en los grupos centrales, viendo FT son fluctuaciones que ocurren en otros datasets.


## Tareas

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

Elegir mejor GMF y NCF

# Gráficas


- Ejecución de todas las agregaciones
- Hacer column-boxplot teniendo la misma escala.


Figura1
Para cada modelo (GMF-MLP)
Para cada función de agregación (min, max, mean, median, mode)
Para cada tamaño de grupo (2-10)
Dos métricas MAE y MSE

GMF



https://stackoverflow.com/questions/45875143/seaborn-making-barplot-by-group-with-asymmetrical-custom-error-bars
https://stackoverflow.com/questions/35978727/how-add-asymmetric-errorbars-to-pandas-grouped-barplot
https://stackoverflow.com/questions/23000418/adding-error-bars-to-grouped-bar-plot-in-pandas



mae-mse
fig1 (mae) gmf min,max
fig2 (mae) mlp min, max, median..
fig1* mse
fig2* mse

mae
fig3 modelos

mse
fig4 modelos 


