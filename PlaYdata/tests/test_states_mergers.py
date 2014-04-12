# -*- coding: utf8 -*-
'''
Created on Apr 10, 2014

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
from PlaYdata.core.base_matrix import StatesMatrixMerger
import PlaYdata.util.tools as tools


def test_StatesMatrixMerger():
    val_mat1_text = list(tools.ngram(test_text_df["text"].values[0], [1]))
    val_mat2_text = list(tools.ngram(test_text_df["text"].values[1], [1]))
    val_mat1 = ValuesMatrix(val_mat1_text, force2d="as_col")
    val_mat2 = ValuesMatrix(val_mat2_text, force2d="as_col")

    idx_data_mat1 = val_mat1.build_index_data_matrix()
    idx_data_mat2 = val_mat2.build_index_data_matrix()
    idx_data_mat1_old_ref_data = idx_data_mat1.states_matrix._ref_data
    idx_data_mat2_old_ref_data = idx_data_mat2.states_matrix._ref_data

    old_idx_data_mat1 = idx_data_mat1.index_matrix.copy()
    old_idx_data_mat2 = idx_data_mat2.index_matrix.copy()

    assert len(idx_data_mat1_old_ref_data) > 0
    assert len(idx_data_mat2_old_ref_data) > 0

    states_mats_merger = StatesMatrixMerger(idx_data_mat1.states_matrix,
                                            idx_data_mat2.states_matrix)

    assert len(states_mats_merger._unique_states_matrix_ids) > 1

    states_mats_merger.update()

    assert len(idx_data_mat1_old_ref_data) == 0
    assert len(idx_data_mat2_old_ref_data) == 0
    assert id(idx_data_mat1.states_matrix) == id(idx_data_mat2.states_matrix)
    assert len(idx_data_mat2.states_matrix._ref_data) > 1

    assert np.array_equal(idx_data_mat1.index_matrix, old_idx_data_mat1)
    assert not np.array_equal(idx_data_mat2.index_matrix, old_idx_data_mat2)

    assert np.array_equal(idx_data_mat1._data, val_mat1)
    assert np.array_equal(idx_data_mat2._data, val_mat2)

if __name__ == '__main__':
    pass
