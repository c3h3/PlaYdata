# -*- coding: utf8 -*-
'''
Created on Apr 7, 2014

@author: c3h3
'''

import numpy as np
# import scipy as sp
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
test_text_df["idx"] = map(lambda xx: u"%05d" % xx, test_text_df["index"])
test_text_df["r1"] = np.random.randn(test_text_df.shape[0])
test_text_df["r2"] = np.random.randn(test_text_df.shape[0])


from PlaYdata.core.base_matrix import Matrix, ValuesMatrix
import PlaYdata.util.tools as tools


def test_Matrix():
    row_mat = Matrix(list(tools.ngram(test_text_df["text"].values[0], [1])),
                     force2d="as_row")

    col_mat = Matrix(list(tools.ngram(test_text_df["text"].values[0], [1])),
                     force2d="as_col")

    assert row_mat.shape == (1, 6)
    assert col_mat.shape == (6, 1)
    assert row_mat._is_1d
    assert col_mat._is_1d


def test_Matrix_building_values_index():
    val_mat1_text = list(tools.ngram(test_text_df["text"].values[0], [1]))
    val_mat2_text = list(tools.ngram(test_text_df["text"].values[1], [1]))
    val_mat1 = ValuesMatrix(val_mat1_text, force2d="as_col")
    val_mat2 = ValuesMatrix(val_mat2_text, force2d="as_col")
    val_mat = ValuesMatrix(np.concatenate((val_mat1._1d_ngram(2),
                                           val_mat2._1d_ngram(2)), axis=0))
    states_matrix, idx_matrix = val_mat.build_values_index()
    assert np.array_equal(states_matrix._eval(idx_matrix), val_mat)


def test_Matrix_build_row_struct_index():
    val_mat1_text = list(tools.ngram(test_text_df["text"].values[0], [1]))
    val_mat2_text = list(tools.ngram(test_text_df["text"].values[1], [1]))
    val_mat1 = ValuesMatrix(val_mat1_text, force2d="as_col")
    val_mat2 = ValuesMatrix(val_mat2_text, force2d="as_col")
    val_mat = ValuesMatrix(np.concatenate((val_mat1._1d_ngram(2), val_mat2._1d_ngram(2)), axis=0))
    states_matrix, idx_matrix = val_mat.build_row_struct_index()
    assert np.array_equal(states_matrix._eval(idx_matrix), val_mat)


def test_Matrix_build_index():
    val_mat1_text = list(tools.ngram(test_text_df["text"].values[0], [1]))
    val_mat2_text = list(tools.ngram(test_text_df["text"].values[1], [1]))
    val_mat1 = ValuesMatrix(val_mat1_text, force2d="as_col")
    val_mat2 = ValuesMatrix(val_mat2_text, force2d="as_col")
    val_mat = ValuesMatrix(np.concatenate((val_mat1._1d_ngram(2),
                                           val_mat2._1d_ngram(2)), axis=0))
    states_matrix, idx_matrix = val_mat.build_values_index()
    assert np.array_equal(states_matrix._eval(idx_matrix), val_mat)
    ss, ii = idx_matrix.build_row_struct_index()
    assert np.array_equal(ss._eval(ii), idx_matrix)
    assert np.array_equal(states_matrix._eval(ss._eval(ii)), val_mat)




if __name__ == '__main__':
    pass
