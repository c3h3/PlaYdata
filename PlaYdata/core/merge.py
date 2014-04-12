'''
Created on Apr 12, 2014

@author: c3h3
'''

import numpy as np
from PlaYdata.util.args_tools import check_type_of_args, check_args
import PlaYdata.util.array_tools as tools
from PlaYdata.core.base_matrix import StatesMatrix, IndexTransformVector
from PlaYdata.core.base_matrix import IndexedDataMatrix, MultiIndexedDataMatrix


class StatesMatrixMerger(list):
    def __init__(self, *states_mats):

        # checking states_mats are all StatesDictionary's instance
        assert check_type_of_args(StatesMatrix, *states_mats)

        # checking states_mats have the same _is_row_struct
        assert len(np.unique(np.array(map(lambda xx:xx._is_row_struct, states_mats)))) == 1
        self._is_row_struct = states_mats[0]._is_row_struct

        # checking states_mats have the same dtype.type
        assert len(np.unique(np.array(map(lambda xx:xx.dtype.type, states_mats)))) == 1
        self._dtype_type = states_mats[0].dtype.type

        # checking states_mats have the same _eval_cls
        assert len(np.unique(np.array(map(lambda xx:xx._eval_cls, states_mats)))) == 1
        self._eval_cls = states_mats[0]._eval_cls

        self._is_already_merged = False
        self._is_already_updated = False

        list.__init__(self, states_mats)

    @property
    def _unique_states_matrix_ids(self):
        return np.unique(np.array(map(id, self)))

    @property
    def merged_index_transform_vectors(self):
        # TODO: havn't implement for the case that self._is_row_struct==True
        # assert not self._is_row_struct

        if not self._is_already_merged:
            self.merge()

        if len(self._unique_states_matrix_ids) == 1:
            return None

        else:
            return self._idx_trans_vectors


    @property
    def merged_states_matrix(self):
        # TODO: havn't implement for the case that self._is_row_struct==True
        # assert not self._is_row_struct

        if not self._is_already_merged:
            self.merge()

        if len(self._unique_states_matrix_ids) == 1:
            return self[0]

        else:
            return self._new_states_matrix

    def merge(self):
        if not self._is_already_merged:
            if len(self._unique_states_matrix_ids) > 1:

                if not self._is_row_struct:
                    convert_to_1d_arrays = map(lambda xx: xx._as_1d_array, self)
                    len_of_1d_arrays = map(len, convert_to_1d_arrays)

                    sector_positions = np.cumsum([0] + len_of_1d_arrays)
                    sector_positions_slices = map(lambda xx: slice(*xx), list(tools.ngram(sector_positions, 2)))

                    join_all_states_arrays = np.concatenate(tuple(convert_to_1d_arrays), axis=0)

                    u, i = np.unique(join_all_states_arrays, return_inverse=True)

                    self._idx_trans_vectors = map(lambda xx: IndexTransformVector(i[xx]), sector_positions_slices)

                    self._new_states_matrix = StatesMatrix(data=u, eval_cls=self._eval_cls)

                # TODO: self._is_row_struct == True
                else:
                    self._is_already_merged = True
                    self._is_already_updated = True
                    pass

            else:
                # Case of len(self._unique_states_matrix_ids) <= 1,
                # it means states_mats has only one kind of states matrix
                # in this case, we don't need to do anything about merge or update
                self._is_already_merged = True
                self._is_already_updated = True

        return self

    def update(self):
        if not self._is_already_merged:
            self.merge()

        if not self._is_already_updated:
            for states_mat, idx_trans_vec in zip(self, self._idx_trans_vectors):
                states_mat.update_all_ref_data(new_states_matrix=self._new_states_matrix,
                                               idx_trans_vec=idx_trans_vec)

        return self


class IndexedDataMatrixMerger(list):

    def __init__(self, *idxed_data_matrixes):

        # checking states_data_arrays are all StatesDataArray's instance
        assert check_type_of_args(IndexedDataMatrix, *idxed_data_matrixes)

        list.__init__(self, idxed_data_matrixes)

    @property
    def states_matrixes(self):
        return map(lambda xx: xx.states_matrix, self)

    @property
    def index_matrixes(self):
        return map(lambda xx: xx.index_matrix, self)

    @property
    def _index_matrixes_ncols(self):
        return map(lambda xx: xx.index_matrix._ncol, self)
    @property
    def _index_matrixes_nrows(self):
        return map(lambda xx: xx.index_matrix._nrow, self)

    @property
    def _rbind_index_matrixes_type(self):
        unique_types_of_index_matrixes = np.unique(map(type, self.index_matrixes))

        # check do self.index_matrixes have the same type ?
        assert len(unique_types_of_index_matrixes) == 1

        return np.unique(map(type, self.index_matrixes))[0]

    def rbind(self, clean_all_old_ref_data=True):
        # checking states_data_arrays has the same _ncol
        assert len(np.unique(self._index_matrixes_ncols)) == 1

        states_matrix_merger = StatesMatrixMerger(*self.states_matrixes)
        states_matrix_merger.update()
        new_index_matrix = self._rbind_index_matrixes_type(np.concatenate(tuple(self.index_matrixes), axis=0))
        new_states_matrix = states_matrix_merger.merged_states_matrix

        if clean_all_old_ref_data:
            new_states_matrix.pop_out_clean_all_ref_data()

        return IndexedDataMatrix(states_matrix=new_states_matrix,
                                 index_matrix=new_index_matrix)

    def cbind_extend_rows(self):
        new_nrows = reduce(lambda x, y: x * y, self._index_matrixes_nrows)
        extend_nrows = map(lambda xx: new_nrows / xx,
                           self._index_matrixes_nrows)

        return MultiIndexedDataMatrix(map(lambda xx: xx[0].extend_rows(xx[1]),
                                          zip(self, extend_nrows)))

    def cbind(self, bind_method="bind_or_extend"):
        assert bind_method in ("bind_or_extend", "force_extend")
        if bind_method == "bind_or_extend":

            # FOR BIND CASE ....
            if len(np.unique(self._index_matrixes_nrows)) == 1:
                return MultiIndexedDataMatrix(self)
            else:
                return self.cbind_extend_rows()

        else:
            return self.cbind_extend_rows()

    def merge(self, merge_type="rbind", **merge_kwargs):
        # rbind <=> axis=0
        # cbind <=> axis=1
        assert merge_type in ("cbind", "rbind")


class MultiIndexedDataMatrixMerger(list):
    def __init__(self, *multi_idxed_data_matrixes):

        # checking states_data_multi_arrays are all StatesDataMultiArrays's instance
        assert check_type_of_args(MultiIndexedDataMatrix,
                                  *multi_idxed_data_matrixes)

        # checking states_data_multi_arrays are all has the same _ncols
        assert len(np.unique(map(lambda xx: tuple(xx._ncols),
                                 multi_idxed_data_matrixes)).tolist()) == 1

        self._ncols = tuple(multi_idxed_data_matrixes[0]._ncols)

        list.__init__(self, multi_idxed_data_matrixes)

    def merge(self):
        zip_cols = zip(*self)
        merge_cols = map(lambda xx: IndexedDataMatrixMerger(*xx).rbind(),
                         zip_cols)

        return MultiIndexedDataMatrix(merge_cols)


if __name__ == '__main__':
    pass
