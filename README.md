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

```
python src/train/ncf-groups-train-all.py --outputdir='experiments' --dataset='data_groups.GroupDataML1M'
```