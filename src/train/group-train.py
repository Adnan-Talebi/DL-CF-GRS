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


fromngroups=4
tongroups=4


# Model creation
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
from src.models.models import get_model_list, get_model, store_model, mlp_agg


for i, group_size in enumerate(range(fromngroups,tongroups+1)):

    train_secuencer = dataset.get_group_train(group_size, BATCH)
    
    individual_model = keras.models.load_model(args.modelFile)
    model = mlp_agg(individual_model, args.k, dataset, args.seed)
    
    model.summary()
    model.compile(
        loss=tf.keras.losses.MeanAbsoluteError(),
        optimizer=keras.optimizers.Adam(lr=0.001)
    )

    history = model.fit(
        train_secuencer,
        validation_data=None,
        epochs=EPOCH,
        verbose=1,
        steps_per_epoch=steps_per_epoch,
    )

    test_secuencer = dataset.get_group_test(group_size, BATCH)
    
    results = model.evaluate(test_secuencer)
    individualmodel_results = individual_model.evaluate(test_secuencer)
    
    store_model(model, history, results, outputdir)