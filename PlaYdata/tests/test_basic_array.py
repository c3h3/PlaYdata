# -*- coding: utf8 -*-
'''
Created on Apr 5, 2014

@author: c3h3
'''


import pandas as pd 
test_text_df = pd.DataFrame([u"今天天氣很好",
                             u"今天天氣很爛",
                             u"我恨它",
                             u"它恨我",
                             u"我愛它",
                             u"它愛我",
                             u"今天很衰",
                             u"日子一天一天過",
                             u"天天刷牙洗臉"])

test_text_df.columns = ["text"]
test_text_df = test_text_df.reset_index()
test_text_df["idx"] = map(lambda xx :u"%05d" % xx,test_text_df["index"])



import PlaYdata.util.array_tools as tools
from PlaYdata.core.basic_arrays import ValuesArray
import numpy as np


def test_ValuesArray_decompose_into_states_ptrs():
    val_array = ValuesArray(list(tools.ngram(test_text_df["text"].values[0],[1])))
    states_array, ptr_array = val_array.decompose_into_states_ptrs()
    
    assert type(val_array) == type(states_array._eval(ptr_array))
    assert np.array_equal(val_array, states_array._eval(ptr_array))
    
    
def test_StatesDictionary__referred_by():
    val_array = ValuesArray(list(tools.ngram(test_text_df["text"].values[0],[1])))
    states_data_array = val_array.to_states_data_array()
    assert states_data_array in states_data_array._states_dict._referred_by










if __name__ == '__main__':
    pass