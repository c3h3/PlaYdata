# -*- coding: utf8 -*-
'''
Created on Apr 7, 2014

@author: c3h3
'''

import numpy as np
import scipy as sp
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
test_text_df["r1"] = np.random.randn(test_text_df.shape[0])
test_text_df["r2"] = np.random.randn(test_text_df.shape[0])


from PlaYdata.core.base_matrix import Matrix
import PlaYdata.util.tools as tools 


def test_Matrix():
    row_mat = Matrix(list(tools.ngram(test_text_df["text"].values[0],[1])),force2d="as_row")
    col_mat = Matrix(list(tools.ngram(test_text_df["text"].values[0],[1])),force2d="as_col")
    assert row_mat.shape == (1,6)
    assert col_mat.shape  == (6,1)
    assert row_mat._is_1d
    assert col_mat._is_1d




if __name__ == '__main__':
    pass