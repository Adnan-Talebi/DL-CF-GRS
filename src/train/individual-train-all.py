from data_utils import init_random
init_random()

import argparse
import sys
import os

from src.models.models import get_model_list, get_model, store_model

model_list = get_model_list()
seed=1234
k = 8

text='Train all models'
parser = argparse.ArgumentParser(description=text)
parser.add_argument('--outdir', type=str, required=True, help="Output directory")
parser.add_argument('--dataset', type=str, required=True, help="Dataset")
args = parser.parse_args()


for model in model_list:
    os.system(f"python src/train/individual-train.py --outdir {args.outdir} --model {model} --seed {seed} --k {k} --dataset {args.dataset}")