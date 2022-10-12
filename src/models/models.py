from data_utils import init_random

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Embedding, Input, Dense, Reshape, Flatten, Concatenate, Multiply
import numpy as np
import matplotlib.pyplot as plt


from keras.regularizers import l2


def mlp(model_name, k, dataset, seed):
    ds_shape, nuser, nitems, ds_code = dataset.get_shape(), dataset.get_num_users(), dataset.get_num_items(), dataset.get_data_code()
    # Do not put a dataset function call in lambda inside a funciton(mlp_2), keras get a reference and try to serialize it when the model is saved.
    # Very weird situation, dificult to debug. It works in the main file but not when put inside a function
    regs=[0,0]
    
    input_layer = layers.Input(shape=ds_shape, name="entrada")

    e_user = layers.Dense(
        k,
        name="emb_u",
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[0])
    )(layers.Lambda(lambda x: x[:, 0:nuser])(input_layer))
    
    e_item = layers.Dense(
        k,
        name="emb_i",
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[1])
    )(layers.Lambda(lambda x: x[:, nuser:])(input_layer))
    
    user_latent = Flatten()(e_user)
    item_latent = Flatten()(e_item)
    
    vector = Concatenate()([user_latent, item_latent])
    # MLP layers
    for idx in [64,32,16,8]:
        layer = Dense(idx, kernel_regularizer= l2(0), activation='relu', name = 'layer%d' %idx)
        vector = layer(vector)
    
    outputs = Dense(1, activation='linear', kernel_initializer='lecun_uniform', name = 'prediction')(vector)
    
    model = keras.Model(inputs=input_layer, outputs=outputs, name=model_name)
    model.summary()
    
    return model


AGG_PREFIX='mlp_agg_'

def mlp_agg(model, k, dataset, seed):
    ds_shape, nuser, nitems, ds_code = dataset.get_shape(), dataset.get_num_users(), dataset.get_num_items(), dataset.get_data_code()
    input_layer = layers.Input(shape=ds_shape, name="entrada")
    regs=[0,0]
    
    e_user = layers.Dense(
        k,
        name="emb_u",
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[0])
    )(layers.Lambda(lambda x: x[:, 0:nuser])(input_layer))

    user_latent = Flatten()(e_user)

    vector = user_latent
    # MLP layers
    for idx in [64,32,16,nuser]:
        layer = Dense(idx, kernel_regularizer= l2(0), activation='relu', name = 'layer%d' %idx)
        vector = layer(vector)

    one_hot_item = layers.Lambda(lambda x: x[:, nuser:])(input_layer)
    multihot_user = vector
    mlp_agg = Concatenate()([multihot_user, one_hot_item])
    model.trainable = False

    agg_mlp_model = model(mlp_agg)

    model = keras.Model(inputs=input_layer, outputs=agg_mlp_model, name=AGG_PREFIX+model.name)
    model.summary()
    
    return model


def store_model(model, history, result, outdir):
    model.save(outdir+'/' + model.name + '.h5')
    
    plt.plot(history.history["loss"])
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    
    plt.savefig(outdir+'/' + model.name + '.png')
    plt.clf()
    
    f = open(outdir+'/' + model.name + '.mae.result', "w")
    f.write(f"{result};Evaluate on test data test loss, test acc")
    f.close()

def get_model(model, k, dataset, seed):
    init_random(seed) # Before build the model, seed is set up
    modelname=f"{model}_k{k}_ds{dataset.get_data_code()}_seed{seed}"
    model = eval(model+"(modelname, k, dataset, seed)")
    return model

def get_model_list():
    #return ["evodeep", "gmf", "mlp", "neumf"]
    #return ["mlp"]
    #return ["gmf"]
    #return ["gmf", "mlp"]
    return ["mlp"]