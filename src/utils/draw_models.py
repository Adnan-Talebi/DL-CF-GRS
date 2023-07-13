from data_utils import init_random
init_random()

import argparse
import sys
import os


OUTDIR='results/graph/'

"""
text = 'Program to draw a model.'
parser = argparse.ArgumentParser(description=text)
parser.add_argument('--out', type=str, required=True, help="Output file")
parser.add_argument('--modelFile', type=str, required=True, help="Model file")

args = parser.parse_args()
"""

# Model creation
import tensorflow as tf
#import tensorflow.keras as keras
#import tensorflow.keras.layers as layers
import keras
import keras.layers as layers

import numpy as np
from src.models.models import get_model_list, get_model, store_model, mlp_agg_dense

"""
Requiere
pip install jaxlib==0.4.7
pip install tensorflow==2.12
pip install pydot
"""

import IPython
from IPython.display import SVG


def save(model, name, show_trainable=False):
    keras.utils.plot_model(
        model,
        to_file=OUTDIR+name+'.png',
        #show_shapes=True,
        #show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=96,
        layer_range=None,
        show_layer_activations=True,
        show_trainable=show_trainable,
    )
    """
    keras.utils.plot_model(
        model,
        to_file=OUTDIR+name+'.svg',
        #show_shapes=True,
        #show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        embed=False,
        layer_range=None,
        show_layer_activations=True,
        show_trainable=show_trainable,
    ) It saves it but throw an exception
    """
    IPython.display.SVG(keras.utils.vis_utils.model_to_dot(model).create(prog='dot', format='svg'))
    
    #IPython.display.SVG(keras.utils.vis_utils.model_to_dot(model).create(prog='dot', format='svg'))
    
    model.save(OUTDIR+name+'.draw.h5')    


"""
Individual MLP
"""
# example: experiments/ml100k/mlp_k8_dsml100k_seed1234.h5
path='experiments/ml100k/mlp_k8_dsml100k_seed1234.h5'
model_ori = keras.models.load_model(path)

"""
#model_ori.summary()
#==================================================================================================
#entrada (InputLayer)            [(None, 2625)]       0                                            
#user_input (Lambda)             (None, 943)          0    
#        entrada[0][0]                    
user_input = layers.Input(shape=model_ori._layers[1].output.shape, name="user_input")
#item_input (Lambda)             (None, 1682)         0           entrada[0][0]                    
item_input = layers.Input(shape=model_ori._layers[2].output.shape, name="item_input")
#emb_u (Dense)                   (None, 8)            7552        user_input[0][0]                 
u=model_ori._layers[3](user_input)
#emb_i (Dense)                   (None, 8)            13464       item_input[0][0]                 
i=model_ori._layers[4](item_input)
#emb_u_flat (Flatten)            (None, 8)            0           emb_u[0][0]                      
u=model_ori._layers[5](u)
#emb_i_flat (Flatten)            (None, 8)            0           emb_i[0][0]                      
i=model_ori._layers[6](i)
#concatenate (Concatenate)       (None, 16)           0           emb_u_flat[0][0]                 
#                                                                 emb_i_flat[0][0]                 
x=model_ori._layers[7]([u,i])
#layer64 (Dense)                 (None, 64)           1088        concatenate[0][0]                
x=model_ori._layers[8](x)
#layer32 (Dense)                 (None, 32)           2080        layer64[0][0]                    
x=model_ori._layers[9](x)
#layer16 (Dense)                 (None, 16)           528         layer32[0][0]                    
x=model_ori._layers[10](x)
#layer8 (Dense)                  (None, 8)            136         layer16[0][0]                    
x=model_ori._layers[11](x)
#prediction (Dense)              (None, 1)            9           layer8[0][0]                     
x=model_ori._layers[12](x)
#==================================================================================================
#Total params: 24,857
#Trainable params: 24,857
#Non-trainable params: 0
"""


#model_ori.summary()
#==================================================================================================
#entrada (InputLayer)            [(None, 2625)]       0                                            
#user_input (Lambda)             (None, 943)          0    
#        entrada[0][0]                    
user_input = layers.Input(shape=model_ori.layers[1].output.shape, name="user_input")
#item_input (Lambda)             (None, 1682)         0           entrada[0][0]                    
item_input = layers.Input(shape=model_ori.layers[2].output.shape, name="item_input")
#emb_u (Dense)                   (None, 8)            7552        user_input[0][0]                 
u=model_ori.layers[3](user_input)
#emb_i (Dense)                   (None, 8)            13464       item_input[0][0]                 
i=model_ori.layers[4](item_input)
#emb_u_flat (Flatten)            (None, 8)            0           emb_u[0][0]                      
u=model_ori.layers[5](u)
#emb_i_flat (Flatten)            (None, 8)            0           emb_i[0][0]                      
i=model_ori.layers[6](i)
#concatenate (Concatenate)       (None, 16)           0           emb_u_flat[0][0]                 
#                                                                 emb_i_flat[0][0]                 
x=model_ori.layers[7]([u,i])
#layer64 (Dense)                 (None, 64)           1088        concatenate[0][0]                
x=model_ori.layers[8](x)
#layer32 (Dense)                 (None, 32)           2080        layer64[0][0]                    
x=model_ori.layers[9](x)
#layer16 (Dense)                 (None, 16)           528         layer32[0][0]                    
x=model_ori.layers[10](x)
#layer8 (Dense)                  (None, 8)            136         layer16[0][0]                    
x=model_ori.layers[11](x)
#predict (Dense)              (None, 1)            9           layer8[0][0]                     
model_ori.layers[12]._name='predict'
x=model_ori.layers[12](x)

#==================================================================================================
#Total params: 24,857
#Trainable params: 24,857
#Non-trainable params: 0

model = keras.Model(
            inputs=[user_input,item_input],
            outputs=x
        )
# 

save(model, 'mlp_individual')






"""
Individual GMF
"""
path='experiments/ml100k/gmf_k8_dsml100k_seed1234.h5'
model_ori = keras.models.load_model(path)
#model_ori.summary()

# Layer (type)                   Output Shape         Param #     Connected to                     
#==================================================================================================
# entrada (InputLayer)           [(None, 2625)]       0           []                               
#                                                                                                  
# user_input (Lambda)            (None, 943)          0           ['entrada[0][0]']                
user_input = layers.Input(shape=model_ori.layers[1].output.shape, name="user_input")
# item_input (Lambda)            (None, 1682)         0           ['entrada[0][0]']                
item_input = layers.Input(shape=model_ori.layers[2].output.shape, name="item_input")
# emb_u (Dense)                  (None, 8)            7552        ['user_input[0][0]']             
u=model_ori.layers[3](user_input)                                                                                
# emb_i (Dense)                  (None, 8)            13464       ['item_input[0][0]']             
i=model_ori.layers[4](item_input)                                                                     
# emb_u_flat (Flatten)           (None, 8)            0           ['emb_u[0][0]']                  
u=model_ori.layers[5](u)                                                               
# emb_i_flat (Flatten)           (None, 8)            0           ['emb_i[0][0]']                  
i=model_ori.layers[6](i)
# dot (Dot)                      (None, 1)            0           ['emb_u_flat[0][0]',             
#                                                                  'emb_i_flat[0][0]']                                                                                                               
model_ori.layers[7]._name='predict'
x=model_ori.layers[7]([u,i])
#==================================================================================================
#Total params: 21,016
#Trainable params: 21,016
#Non-trainable params: 0

model = keras.Model(
            inputs=[user_input,item_input],
            outputs=x
        )

save(model, 'gmf_individual')


"""
MLP GMF
"""
path='experiments/ml100k/mlp_agg_dense_gmf_k8_dsml100k_seed1234_2_mean.h5'
model_ori = keras.models.load_model(path)
#model_ori.summary()

#__________________________________________________________________________________________________
#Layer (type)                   Output Shape         Param #     Connected to                     
#==================================================================================================
#entrada (InputLayer)           [(None, 2625)]       0           []                               
#                                                                                                  
#lambda (Lambda)                (None, 943)          0           ['entrada[0][0]']                
user_input = layers.Input(shape=model_ori.layers[1].output.shape, name="multihot_group_input")
#mlp_agg_dense_emb_u (Dense)    (None, 8)            7552        ['lambda[0][0]']                 
u=model_ori.layers[2](user_input)
#flatten (Flatten)              (None, 8)            0           ['mlp_agg_dense_emb_u[0][0]']    
u=model_ori.layers[3](u)
#mlp_agg_dense_layer64 (Dense)  (None, 64)           576         ['flatten[0][0]']                
gu = model_ori.layers[4](u)
#mlp_agg_dense_layer32 (Dense)  (None, 32)           2080        ['mlp_agg_dense_layer64[0][0]']  
gu = model_ori.layers[5](gu)
#lambda_1 (Lambda)              (None, 1682)         0           ['entrada[0][0]']                
item_input = layers.Input(shape=model_ori.layers[6].output.shape, name="item_input")
#mlp_agg_dense_layer16 (Dense)  (None, 16)           528         ['mlp_agg_dense_layer32[0][0]']  
gu = model_ori.layers[7](gu)
#emb_i (Dense)                  (None, 8)            13464       ['lambda_1[0][0]']               
i=model_ori.layers[8](item_input)
#mlp_agg_dense_layer8 (Dense)   (None, 8)            136         ['mlp_agg_dense_layer16[0][0]']  
gu = model_ori.layers[9](gu)
#emb_i_flat (Flatten)           (None, 8)            0           ['emb_i[0][0]']                  
i=model_ori.layers[10](i)
#dot (Dot)                      (None, 1)            0           ['mlp_agg_dense_layer8[0][0]',   
                                                                #'emb_i_flat[0][0]']  
#                                                                  
model_ori.layers[11]._name='predict'
x=model_ori.layers[11]([gu,i])
model = keras.Model(
            inputs=[user_input,item_input],
            outputs=x
        )

save(model, 'mlp_groups_gmf', True)


"""
MLP MLP
"""
path='experiments/ml100k/mlp_agg_dense_mlp_k8_dsml100k_seed1234_2_mean.h5'
model_ori = keras.models.load_model(path)
#model_ori.summary()

#Model: "mlp_agg_dense_mlp_k8_dsml100k_seed1234_2_mean"
#__________________________________________________________________________________________________
#Layer (type)                   Output Shape         Param #     Connected to                     
#==================================================================================================
#entrada (InputLayer)           [(None, 2625)]       0           []                               
#                                                                                                
#lambda (Lambda)                (None, 943)          0           ['entrada[0][0]']                
user_input = layers.Input(shape=model_ori.layers[1].output.shape, name="multihot_group_input")
#mlp_agg_dense_emb_u (Dense)    (None, 8)            7552        ['lambda[0][0]']                 
u=model_ori.layers[2](user_input)
#flatten (Flatten)              (None, 8)            0           ['mlp_agg_dense_emb_u[0][0]']    
u=model_ori.layers[3](u)
#mlp_agg_dense_layer64 (Dense)  (None, 64)           576         ['flatten[0][0]']                
gu = model_ori.layers[4](u)
#mlp_agg_dense_layer32 (Dense)  (None, 32)           2080        ['mlp_agg_dense_layer64[0][0]']  
gu = model_ori.layers[5](gu)
#lambda_1 (Lambda)              (None, 1682)         0           ['entrada[0][0]']                
item_input = layers.Input(shape=model_ori.layers[6].output.shape, name="item_input")
#mlp_agg_dense_layer16 (Dense)  (None, 16)           528         ['mlp_agg_dense_layer32[0][0]']  
gu = model_ori.layers[7](gu)
#emb_i (Dense)                  (None, 8)            13464       ['lambda_1[0][0]']               
i=model_ori.layers[8](item_input)
#mlp_agg_dense_layer8 (Dense)   (None, 8)            136         ['mlp_agg_dense_layer16[0][0]']  
gu = model_ori.layers[9](gu)
#emb_i_flat (Flatten)           (None, 8)            0           ['emb_i[0][0]']                  
i=model_ori.layers[10](i)
#concatenate (Concatenate)      (None, 16)           0           ['mlp_agg_dense_layer8[0][0]',   
                                                                #'emb_i_flat[0][0]']             
x=model_ori.layers[11]([gu,i]) 
#layer64 (Dense)                (None, 64)           1088        ['concatenate[0][0]']            
x=model_ori.layers[12](x)
#layer32 (Dense)                (None, 32)           2080        ['layer64[0][0]']                
x=model_ori.layers[13](x)
#layer16 (Dense)                (None, 16)           528         ['layer32[0][0]']                
x=model_ori.layers[14](x)
#layer8 (Dense)                 (None, 8)            136         ['layer16[0][0]']                
x=model_ori.layers[15](x)
#prediction (Dense)             (None, 1)            9           ['layer8[0][0]']                 
model_ori.layers[16]._name='predict'
x=model_ori.layers[16](x)
#==================================================================================================
#Total params: 28,177
#Trainable params: 10,872
#Non-trainable params: 17,305

model = keras.Model(
            inputs=[user_input,item_input],
            outputs=x
        )
save(model, 'mlp_groups_mlp', True)