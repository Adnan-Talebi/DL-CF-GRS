import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import pandas as pd

def agg_list():
    """
    return [
        'min',
        'max',
        'mean',
        'median',
        'mode',
    ]
    """
    return [
        'mean',
    ]


def get_agg_function(fname):
    assoc = {
        'min': pd.DataFrame.min,
        'max': pd.DataFrame.max,
        'mean': pd.DataFrame.mean,
        'median': pd.DataFrame.median,
        'mode': pd.DataFrame.mode,
    }
    return assoc[fname]
