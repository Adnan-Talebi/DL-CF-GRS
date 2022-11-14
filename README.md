# dl-cf-groups-deep-aggregation
dl-cf-groups-deep-aggregation

## Data

```
PYTHONPATH="/workspace/rs-data-python:."
export PYTHONPATH
```

## Project Layout

```txt

- src
  \- data 
  \- eval
  \- models
  \- train
  \- utils
```

## Execution example

### Group generation

```
> python
>>> from src.data.data import generate_groups('ml1m')
```

```
python -c "from src.data.data import generate_groups; generate_groups('ml100k')"
```

#### Split Test-Train

```
bash src/data/groups-spliter.sh data/grupos/ml1m
```

### Train MLP with individual data

```
python src/train/individual-train.py  --outdir 'experiments' --model mlp --seed 1234 --k 8 --dataset 'src.data.data.GroupDataFT'
python src/train/individual-train.py  --outdir 'experiments' --model mlp --seed 1234 --k 8 --dataset 'src.data.data.GroupDataML1M'
python src/train/individual-train.py  --outdir 'experiments' --model mlp --seed 1234 --k 8 --dataset 'src.data.data.GroupDataANIME'
```

### Train Aggregator as MLP

```
python src/train/group-train-all.py  --outdir 'experiments' --model 'experiments/ft/mlp_k8_dsft_seed1234.h5' --seed 1234 --k 8 --dataset 'src.data.data.GroupDataFT'
python src/train/group-train-all.py  --outdir 'experiments' --model 'experiments/ml1m/mlp_k8_dsml1m_seed1234.h5' --seed 1234 --k 8 --dataset 'src.data.data.GroupDataML1M'
```

### Eval

```
python src/eval/eval.py  --outdir 'results' --modelFile 'experiments/ft/mlp_k8_dsft_seed1234.h5' --seed 1234 --k 8 --dataset 'src.data.data.GroupDataFT'
python src/eval/eval.py  --outdir 'results' --modelFile 'experiments/ml1m/mlp_k8_dsml1m_seed1234.h5' --seed 1234 --k 8 --dataset 'src.data.data.GroupDataML1M'
```

## Tareas

(x) Aumentar tamaño de grupos. Mantener grupos generados de test.
(x) Pegar salida a embedding del modelo individual
(x) Generar Test-Train y usar validation
(x) Poner early stop y aumentar el número de EPOCHs

Tamaño de lote divisor de la muestra
Probar entrenar: MAX, MIN, MEAN, MEDIAN


Meter la moda.
Meter NCF y GMF con softmax
Sacar info de GMF, NCF con variaciones
Elegir mejor GMF y NCF
Pintar STD