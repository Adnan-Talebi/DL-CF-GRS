from data_utils import init_random
init_random()

import argparse
import sys
import os

text = 'Program to train a model with groups data. It uses a Dense as an embedding.'
parser = argparse.ArgumentParser(description=text)
parser.add_argument('--outdir', type=str, required=True, help="Output directory")
parser.add_argument('--modelFile', type=str, required=True, help="Model file")
parser.add_argument('--seed', type=str, required=True, help="Seed")
parser.add_argument('--k', type=int, required=True, help="Number of factor in each embedding")
parser.add_argument('--dataset', type=str, required=True, help="Dataset")
parser.add_argument('--agg', type=str, required=False, help="Agg")

args = parser.parse_args()

from src.utils.agg_functions import agg_list
aggs = agg_list()

if(args.agg!="" and args.agg!=None):
    print("Concrete aggregation" + args.agg)
    aggs = [args.agg]

fromngroups=2
tongroups=10

for i, group_size in enumerate(range(fromngroups,tongroups+1)):
    for agg in aggs:
        os.system(f"python src/train/group-train.py  --outdir {args.outdir} --model {args.modelFile} --seed {args.seed} --k {args.k} --dataset {args.dataset} --g {group_size} --agg {agg}")