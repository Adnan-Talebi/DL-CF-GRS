from data_utils import init_random
init_random()

import argparse
import sys
import os
from pathlib import Path

text = 'Program to train a model with groups data. It uses a Dense as an embedding.'
parser = argparse.ArgumentParser(description=text)
parser.add_argument('--outdir', type=str, required=True, help="Output directory")
parser.add_argument('--modelFile', type=str, required=True, help="Model file")
parser.add_argument('--modelName', type=str, required=True, help="Model name for outputfile")
parser.add_argument('--seed', type=str, required=True, help="Seed")
parser.add_argument('--k', type=int, required=True, help="Number of factor in each embedding")
parser.add_argument('--dataset', type=str, required=True, help="Dataset")

args = parser.parse_args()


BATCH=64
EPOCH=25
steps_per_epoch = None

# NETFLIX
#EPOCH=5
#steps_per_epoch = 200000

# Seed inizialization
init_random(args.seed)

# Dataset load
from data_utils import dynamic_import
from data_groups import OneHotGenerator

DynamicClass = dynamic_import(args.dataset)
dataset = DynamicClass()


# Output directory creation
outputdir=args.outdir+"/"+dataset.get_data_code()

if not os.path.exists(outputdir):
    os.makedirs(outputdir)
    print("The new directory %s is created!" % outputdir)


"""
Write prediction (data) in file 
"""
def write_file(dir, gsize, modelname, data, index=True):
    f=open(f'{dir}/groups-{gsize}-{modelname}.csv', 'w')
    f.write(f'{dir}/groups-{gsize}-{modelname}'+"\n")
    for r in data:
        if index:
            f.write(str(r[0])+"\n")
        else:
            f.write(str(r)+"\n")
    f.close()

fromngroups=2
tongroups=10


# Model creation
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
from src.models.models import get_model_list, get_model, store_model

individual_model = keras.models.load_model(args.modelFile)
outdir = outputdir

from src.utils.agg_functions import agg_list
aggs = agg_list()
modelname_outputfile = args.modelName

from multihot_activations import get_activation_expert, get_activation_softmax
import pandas as pd

for i, group_size in enumerate(range(fromngroups,tongroups+1)):
    """
        IPA. Mean of individuals prediction
    """    
    test_secuencer_as_individuals = dataset.get_group_test_as_individuals(group_size, BATCH)
    y_pred = individual_model.predict(test_secuencer_as_individuals)
    y_pred = y_pred.reshape((-1, group_size))
    y_pred = np.mean(y_pred, axis=1)
    y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    y_pred[y_pred<dataset.get_rating_min()]=dataset.get_rating_min()
    # [[predict_rating1, predict_rating2, ..., predict_ratingn], [predict_rating1, ...] ... ]
    write_file(outdir, group_size, modelname_outputfile + '_ipa', y_pred, index=False)
    
    """
        model file name, path, etc.
    """
    base_name = args.modelFile.split(".")[0] # remove extension
    pathtomodel = base_name.split('/')       # a / b / c
    modelname = pathtomodel.pop()            # a,b    c
    pathtomodel = "/".join(pathtomodel)      # a / b
    
    """
        Aggregation in MLP: gpa_mlp_<func>
    """
    for agg in aggs:
        test_secuencer = dataset.get_group_test(group_size, BATCH, agg, 1.0) # None = 1/grpsize
        
        group_model_dense = keras.models.load_model(pathtomodel+"/mlp_agg_dense_"+modelname+"_"+str(group_size)+"_"+agg+".h5")
        y_pred = group_model_dense.predict(test_secuencer)
        y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
        y_pred[y_pred<dataset.get_rating_min()]=dataset.get_rating_min()
        write_file(outdir, group_size, modelname_outputfile + '_gpa_mlp_'+agg, y_pred)
    
            
    """
        Baselines gpa_avg, gpa_expert, gpa_softmax
    """
    test_data = dataset.get_group_test(group_size, BATCH, None, None) # No agg is used. Activation 1/grpsize
    y_pred = individual_model.predict(test_data)
    y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    y_pred[y_pred<dataset.get_rating_min()]=dataset.get_rating_min()
    write_file(outdir,group_size, modelname_outputfile+"_gpa_avg", y_pred)
    
    
    """
        Expert
    """
    expert_closure = get_activation_expert(
                        dataset.get_num_users(),
                        dataset.get_num_items(),
                        group_size,
                        dataset.get_rating_count()
                    )
    test_data = dataset.get_group_test(group_size, BATCH, None, expert_closure)
    y_pred = individual_model.predict(test_data)
    y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    y_pred[y_pred<dataset.get_rating_min()]=dataset.get_rating_min()
    write_file(outdir,group_size, modelname_outputfile+"_gpa_expert", y_pred)
    
    
    """
        Softmax
    """
    softmax_closure = get_activation_softmax(
                        dataset.get_num_users(),
                        dataset.get_num_items(),
                        group_size,
                        dataset.get_rating_count()
                    )
    test_data = dataset.get_group_test(group_size, BATCH, None, softmax_closure)
    y_pred = individual_model.predict(test_data)
    y_pred[y_pred>dataset.get_rating_max()]=dataset.get_rating_max()
    y_pred[y_pred<dataset.get_rating_min()]=dataset.get_rating_min()
    write_file(outdir,group_size, modelname_outputfile+"_gpa_softmax", y_pred)
