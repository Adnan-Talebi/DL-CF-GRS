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
parser.add_argument('--group_size', type=int, required=True, help="group_size")
parser.add_argument('--agg', type=str, required=True, help="Agg function")

args = parser.parse_args()


BATCH=64
EPOCH=1000
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





# Model creation
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
from src.models.models import get_model_list, get_model, store_model, mlp_agg, mlp_agg_dense


group_size = args.group_size

from src.utils.agg_functions import get_agg_function
agg_function = get_agg_function(args.agg)

train_secuencer = dataset.get_group_train(group_size, BATCH, agg_function)
val_secuencer = dataset.get_group_val(group_size, BATCH, agg_function)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

individual_model = keras.models.load_model(args.modelFile)
individual_model.trainable = False

model_agg = mlp_agg(individual_model, args.k, dataset, args.seed)
model_agg._name = model_agg._name + '_' + str(group_size) + '_' + args.agg

model_agg.summary()
model_agg.compile(
    loss=tf.keras.losses.MeanAbsoluteError(),
    optimizer=keras.optimizers.Adam(lr=0.001)
)

history = model_agg.fit(
    train_secuencer,
    validation_data=val_secuencer,
    epochs=EPOCH,
    verbose=1,
    callbacks=[callback],
    steps_per_epoch=steps_per_epoch,
)

model_agg_dense = mlp_agg_dense(individual_model, args.k, dataset, args.seed)
model_agg_dense._name = model_agg_dense._name + '_' + str(group_size) + '_' + args.agg

model_agg_dense.summary()
model_agg_dense.compile(
    loss=tf.keras.losses.MeanAbsoluteError(),
    optimizer=keras.optimizers.Adam(lr=0.001)
)

history = model_agg_dense.fit(
    train_secuencer,
    validation_data=val_secuencer,
    epochs=EPOCH,
    verbose=1,
    callbacks=[callback],
    steps_per_epoch=steps_per_epoch,
)


test_secuencer = dataset.get_group_test(group_size, BATCH, agg_function)

print("MLP as AGG MultiHot")
results = model_agg.evaluate(test_secuencer)
store_model(model_agg, history, results, outputdir)

print("MLP as AGG Dense")
results = model_agg_dense.evaluate(test_secuencer)
store_model(model_agg_dense, history, results, outputdir)

print("MLP individual")
individualmodel_results = individual_model.evaluate(test_secuencer)
    