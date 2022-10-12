import os
import math

import numpy as np
import pandas as pd
from keras.utils import Sequence

from data_general import GeneralRS
from data_general_mixin import ReadTestMixin, ReadTrainMixin, GetMaxMinRatingMixin, ShuffleMixin, SplitValMixin, GetNumsByMaxMixin
from data_onehot import dense_embedding, dense_embedding_for_group, dense_embedding_for_group_with_expert_info, dense_embedding_for_group_max, dense_embedding_for_group_min, dense_embedding_for_group_with_inverse_expert_info, dense_embedding_for_group_with_raro, dense_embedding_for_group_with_softmax, dense_embedding_for_group_with_inverse_softmax, dense_embedding_for_group_with_softmax_raro, dense_embedding_for_group_with_item
from data_groups import MultiHotGenerator


DATA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/data"


class GroupData(ReadTestMixin, ReadTrainMixin, ShuffleMixin, SplitValMixin, GetMaxMinRatingMixin, GetNumsByMaxMixin, GeneralRS, object):

    def __init__(self, *args, **kwargs):
        super(GroupData, self).__init__(*args, **kwargs)
    
    def get_shape(self):
        return (self.get_num_users() + self.get_num_items(),) # OneHot Dense as Embedding
    
    def __get_generator(self, group_size, batch_size, path):
        return MultiHotGenerator(
                    group_size,
                    path,
                    self.get_num_users(),
                    self.get_num_items(),
                    batch_size
                )
        
    def get_group_train(self, group_size, batch_size):
        return self.__get_generator(group_size, batch_size, DATA_ROOT+"/grupos/" + self.get_data_code() + "/groups-"+str(group_size)+"-train.csv")
        
    def get_group_test(self, group_size, batch_size):
        return self.__get_generator(group_size, batch_size, DATA_ROOT+"/grupos/" + self.get_data_code() + "/groups-"+str(group_size)+"-test.csv")
    
    def get_group_test_as_individuals(self, group_size, batch_size):
        return OneHotGeneratorAsIndividual(
                    group_size,
                    DATA_ROOT+"/grupos/" + self.get_data_code() + "/groups-"+str(group_size)+"-test.csv",
                    self.get_num_users(),
                    self.get_num_items(),
                    batch
                )


class GroupDataML(GroupData):
    test_url = "/grupos/ml/test-ratings.csv"
    train_url = "/grupos/ml/training-ratings.csv"
    code = "ml"


class GroupDataML1M(GroupData):
    test_url = "/grupos/ml1m/test-ratings.csv"
    train_url = "/grupos/ml1m/training-ratings.csv"
    code = "ml1m"


class GroupDataML1MCompleteInfo(GroupData):
    test_url = "/grupos/ml1m-completeinfo/test-ratings.csv"
    train_url = "/grupos/ml1m-completeinfo/training-ratings.csv"
    code = "ml1m-completeinfo"


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


class SimpleGroupDataNetflix(ReadTestMixin, ReadTrainMixin, GetMaxMinRatingMixin, GetNumsByMaxMixin, GeneralRS, object):
    test_url = "/grupos/netflix/test-ratings.csv"
    train_url = "/grupos/netflix/training-ratings.csv"
    code = "simplenetflix"

if __name__ == '__main__':
    test = SimpleGroupDataNetflix()
    print(test.info())