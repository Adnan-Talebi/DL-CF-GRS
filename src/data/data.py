import os
import math

import numpy as np
import pandas as pd
from keras.utils import Sequence

from data_general import GeneralRS
from data_general_mixin import ReadTestMixin, ReadTrainMixin, ExpertInfoMixin, GetMaxMinRatingMixin, ShuffleMixin, SplitValMixin, GetNumsByMaxMixin
from data_onehot import dense_embedding, dense_embedding_for_group
from data_groups import MultiHotGenerator, OneHotGeneratorAsIndividual


DATA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/data"


class GroupData(ReadTestMixin, ReadTrainMixin, ExpertInfoMixin, ShuffleMixin, SplitValMixin, GetMaxMinRatingMixin, GetNumsByMaxMixin, GeneralRS, object):

    def __init__(self, *args, **kwargs):
        super(GroupData, self).__init__(*args, **kwargs)
    
    def get_shape(self):
        return (self.get_num_users() + self.get_num_items(),) # OneHot Dense as Embedding
    
    def get_data_root(self):
        return DATA_ROOT
    
    def __get_generator(self, group_size, batch_size, path, agg_function, activation_function):
        return MultiHotGenerator(
                    group_size,
                    path,
                    self.get_num_users(),
                    self.get_num_items(),
                    batch_size,
                    agg_function,
                    activation_function
                )
        
    def get_group_train(self, group_size, batch_size, agg_function, activation_function):
        return self.__get_generator(group_size, batch_size, DATA_ROOT+"/grupos/" + self.get_data_code() + "/groups-"+str(group_size)+"-train.csv", agg_function, activation_function)
    
    def get_group_val(self, group_size, batch_size, agg_function, activation_function):
        return self.__get_generator(group_size, batch_size, DATA_ROOT+"/grupos/" + self.get_data_code() + "/groups-"+str(group_size)+"-val.csv", agg_function, activation_function)
    
    def get_group_test(self, group_size, batch_size, agg_function, activation_function):
        return self.__get_generator(group_size, batch_size, DATA_ROOT+"/grupos/" + self.get_data_code() + "/groups-"+str(group_size)+"-test.csv", agg_function, activation_function)
    
    def get_group_test_as_individuals(self, group_size, batch_size):
        return OneHotGeneratorAsIndividual(
                    group_size,
                    DATA_ROOT+"/grupos/" + self.get_data_code() + "/groups-"+str(group_size)+"-test.csv",
                    self.get_num_users(),
                    self.get_num_items(),
                    batch_size
                )


def generate_group(data, data_code, group_size):
    rating_count = data.groupby('i',as_index=False).count()
    filtered = rating_count[rating_count['u']>group_size]
    # Number of groups depends on dataset and groups size
    samples = filtered.sample(len(data)*group_size,replace=True)

    groups_data = []
    for item in samples['i'].tolist():
        item_data = data[data['i']==item].sample(group_size)
        group_row = [item]
        for idx, row in item_data.iterrows():
            group_row.append(int(row['u']))
            group_row.append(row['r'])
        groups_data.append(group_row)
    
    header = ['item']
    for u in range(0, group_size):
        header.append('user-'+str(u+1))
        header.append('rating-'+str(u+1))
    
    groups = pd.DataFrame(groups_data, columns=header, dtype=int)

    # Read test
    datapathout = DATA_ROOT+"/grupos/" + data_code + f"/groups-{group_size}.csv"
    groups.to_csv(datapathout, index=False)


"""
    Recive a dataset from rs-data-python
"""
def generate_groups(data_code):
    fromngroups=2
    tongroups=10
    for i, group_size in enumerate(range(fromngroups,tongroups+1)):         
        data = pd.read_csv(
                DATA_ROOT+"/grupos/" + data_code + "/test-ratings.csv",
                header=0,
                names=['u','i','r'],
        )
        generate_group(data, data_code, group_size)


class GroupDataML1M(GroupData):
    test_url = "/grupos/ml1m/test-ratings.csv"
    train_url = "/grupos/ml1m/training-ratings.csv"
    code = "ml1m"


class GroupDataFT(GroupData):
    test_url = "/grupos/ft/test-ratings.csv"
    train_url = "/grupos/ft/training-ratings.csv"
    code = "ft"


class GroupDataANIME(GroupData):
    test_url = "/grupos/anime/test-ratings.csv"
    train_url = "/grupos/anime/training-ratings.csv"
    code = "anime"


class GroupDataNetflix(GroupData):
    test_url = "/grupos/netflix/test-ratings.csv"
    train_url = "/grupos/netflix/training-ratings.csv"
    code = "netflix"


def code_to_py(code):
    if code == 'ft':
        return 'src.data.data.GroupDataFT'
    if code == 'ml1m':
        return 'src.data.data.GroupDataML1M'
    if code == 'anime':
        return 'src.data.data.GroupDataANIME'
