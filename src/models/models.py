from data_utils import init_random

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Embedding, Input, Dense, Reshape, Flatten, Concatenate, Dot
import numpy as np
import matplotlib.pyplot as plt


from keras.regularizers import l2
regs=[0.0001,0.0001]

def mlp(model_name, k, dataset, seed):
    ds_shape, nuser, nitems, ds_code = dataset.get_shape(), dataset.get_num_users(), dataset.get_num_items(), dataset.get_data_code()
    # Do not put a dataset function call in lambda inside a funciton(mlp_2), keras get a reference and try to serialize it when the model is saved.
    # Very weird situation, dificult to debug. It works in the main file but not when put inside a function
    
    input_layer = layers.Input(shape=ds_shape, name="entrada")

    e_user = layers.Dense(
        k,
        name="emb_u",
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[0])
    )(layers.Lambda(lambda x: x[:, 0:nuser], name='user_input')(input_layer))
    
    e_item = layers.Dense(
        k,
        name="emb_i",
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[1])
    )(layers.Lambda(lambda x: x[:, nuser:], name='item_input')(input_layer))
    
    user_latent = Flatten(name="emb_u_flat")(e_user)
    item_latent = Flatten(name="emb_i_flat")(e_item)
    
    vector = Concatenate(name='concatenate')([user_latent, item_latent])
    # MLP layers
    for idx in [64,32,16,8]:
        layer = Dense(idx, kernel_regularizer= l2(regs[0]), activation='relu', name = 'layer%d' %idx)
        vector = layer(vector)
    
    outputs = Dense(1, activation='linear', kernel_initializer='lecun_uniform', name = 'prediction')(vector)
    
    model = keras.Model(inputs=input_layer, outputs=outputs, name=model_name)
    model.summary()
    
    return model


def gmf(model_name, k, dataset, seed):
    ds_shape, nuser, nitems, ds_code = dataset.get_shape(), dataset.get_num_users(), dataset.get_num_items(), dataset.get_data_code()
    # Do not put a dataset function call in lambda inside a funciton(mlp_2), keras get a reference and try to serialize it when the model is saved.
    # Very weird situation, dificult to debug. It works in the main file but not when put inside a function

    input_layer = layers.Input(shape=ds_shape, name="entrada")

    e_user = layers.Dense(
        k,
        name="emb_u",
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[0])
    )(layers.Lambda(lambda x: x[:, 0:nuser], name='user_input')(input_layer))

    e_item = layers.Dense(
        k,
        name="emb_i",
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[1])
    )(layers.Lambda(lambda x: x[:, nuser:], name='item_input')(input_layer))

    # Crucial to flatten an embedding vector!
    user_latent = Flatten(name="emb_u_flat")(e_user)
    item_latent = Flatten(name="emb_i_flat")(e_item)

    # Element-wise product of user and item embeddings
    #predict_vector = merge([user_latent, item_latent], mode = 'mul')
    predict_vector = Dot(name='dot', axes=1)([user_latent, item_latent])

    # Final prediction layer
    #prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    #prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(predict_vector)
    # CHANGED to linear for regression
    #outputs = Dense(1, activation='linear', kernel_initializer='lecun_uniform', name = 'prediction')(predict_vector)
    outputs = predict_vector
    
    model = keras.Model(inputs=input_layer, outputs=outputs, name=model_name)
    model.summary()

    return model


AGG_PREFIX_D='mlp_agg_dense_'
def mlp_agg_dense(model, k, dataset, seed):
    if 'gmf_' in model.name:
        return mlp_agg_dense_to_gmf(model, k, dataset, seed)
    else:
        return mlp_agg_dense_to_mlp(model, k, dataset, seed)

def mlp_for_groups(ds):
    # MLP layers
    # ML100K y FT -> [64, 32, 16, 8]
    if 'ml100k' in ds.get_data_code() or 'ft' in ds.get_data_code():
        return [2**i for i in range(6,2,-1)]
    # ML1M -> [128, 64, 32, 16, 8]
    if 'ml1m' in ds.get_data_code() or 'anime' in ds.get_data_code():#JDL: 2023-0329 try anime with less layers
        return [2**i for i in range(7,2,-1)]
    # ANIME -> [256, 128, 64, 32, 16, 8]
    #if 'anime' in ds.get_data_code():
    #    return [2**i for i in range(8,2,-1)]


def mlp_agg_dense_to_mlp(model, k, dataset, seed):
    ds_shape, nuser, nitems, ds_code = dataset.get_shape(), dataset.get_num_users(), dataset.get_num_items(), dataset.get_data_code()
    
    """ MLP aggregation for group """
    input_layer = layers.Input(shape=ds_shape, name="entrada")
    
    e_user = layers.Dense(
        k,
        name=AGG_PREFIX_D+"emb_u",
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[0])
    )(layers.Lambda(lambda x: x[:, 0:nuser])(input_layer))

    user_latent = Flatten()(e_user)

    vector = user_latent

    for idx in mlp_for_groups(dataset):
        layer = Dense(idx, kernel_regularizer= l2(regs[0]), activation='relu', name = AGG_PREFIX_D+'layer%d' %idx)
        vector = layer(vector)
    
    # Connect to dense instead of go wide to narrow again
    # multihot_user = Dense(nuser, kernel_regularizer= l2(0), activation='relu', name='agg_as_MLP')(vector)
    #mlp_agg = Concatenate()([group_embedding, one_hot_item])
    group_embedding = vector
    
    """ Pretrained model """
    model.trainable = False

    """
    user_input
    emb_u
    emb_u_flat
    item_input
    emb_i
    emb_i_flat
    concatenacion
    layerX
    prediction
    """
    
    one_hot_item = layers.Lambda(lambda x: x[:, nuser:])(input_layer)
    item_x = model.get_layer('emb_i')(one_hot_item)
    item_x = model.get_layer('emb_i_flat')(item_x)
    mlp_agg = Concatenate()([group_embedding, item_x])
    
    for idx in [64,32,16,8]:
        mlp_agg = model.get_layer('layer%d'%idx)(mlp_agg)
    
    output_layer = model.get_layer('prediction')(mlp_agg)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer, name=AGG_PREFIX_D+model.name)
    model.summary()
    
    return model


def mlp_agg_dense_to_gmf(model, k, dataset, seed):
    ds_shape, nuser, nitems, ds_code = dataset.get_shape(), dataset.get_num_users(), dataset.get_num_items(), dataset.get_data_code()
    
    """ MLP aggregation for group """
    input_layer = layers.Input(shape=ds_shape, name="entrada")
    
    e_user = layers.Dense(
        k,
        name=AGG_PREFIX_D+"emb_u",
        kernel_initializer = 'normal', kernel_regularizer = l2(regs[0])
    )(layers.Lambda(lambda x: x[:, 0:nuser])(input_layer))

    user_latent = Flatten()(e_user)

    vector = user_latent
    # MLP layers
    for idx in mlp_for_groups(dataset):
        layer = Dense(idx, kernel_regularizer= l2(regs[0]), activation='relu', name = AGG_PREFIX_D+'layer%d' %idx)
        vector = layer(vector)
    
    # Connect to dense instead of go wide to narrow again
    # multihot_user = Dense(nuser, kernel_regularizer= l2(0), activation='relu', name='agg_as_MLP')(vector)
    #mlp_agg = Concatenate()([group_embedding, one_hot_item])
    group_embedding = vector
    
    """ Pretrained model """
    model.trainable = False

    """
    user_input
    emb_u
    emb_u_flat
    item_input
    emb_i
    emb_i_flat
    dot
    """
    one_hot_item = layers.Lambda(lambda x: x[:, nuser:])(input_layer)
    item_x = model.get_layer('emb_i')(one_hot_item)
    item_x = model.get_layer('emb_i_flat')(item_x)
    """
    >>> x
    array([[1, 2, 3],
        [4, 5, 6]])
    >>> y
    array([[3, 4, 5],
        [6, 7, 8]])
    >>> tf.keras.layers.Dot(axes=1)([x,y])
    <tf.Tensor: shape=(2, 1), dtype=int64, numpy=
    array([[ 26],
        [107]])>
    """
    predict_vector = Dot(name='dot', axes=1)([group_embedding, item_x])
    output_layer = predict_vector
    
    model = keras.Model(inputs=input_layer, outputs=output_layer, name=AGG_PREFIX_D+model.name)
    model.summary()
    
    return model


def store_model(model, history, result, outdir, extra = ""):
    model.save(outdir+'/' + model.name + extra + '.h5')
    
    plt.plot(history.history["loss"])
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    
    plt.savefig(outdir+'/' + model.name + extra + '.png')
    plt.clf()
    
    f = open(outdir+'/' + model.name + extra + '.mae.result', "w")
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
    return ["gmf", "mlp"]