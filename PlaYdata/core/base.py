'''
Created on Apr 7, 2014

@author: c3h3
'''

import numpy as np
from PlaYdata.util.args_tools import check_type_of_args
import PlaYdata.util.array_tools as tools
import PlaYdata.util.np_tools as np_tools


class Vector(np.ndarray):

    @classmethod
    def _preprocess_if_data_is_cls(cls, vector, *args, **kwargs):
        pass

    @classmethod
    def _preprocess_before_init_return(cls, vector, *args, **kwargs):
        pass

    def __new__(cls, data, dtype=None, *args, **kwargs):

        if isinstance(data, cls):
            values_vector = data
            cls._preprocess_if_data_is_cls(vector=values_vector,
                                           *args, **kwargs)
            return values_vector

        else:
            vector_data = np.array(data, dtype=dtype)

            if len(vector_data.shape) > 1:
                vector_data = np_tools.clean_no_data_tensors(vector_data)

            assert len(vector_data.shape) == 1

            vector_data = vector_data.view(cls)
            cls._preprocess_before_init_return(vector=vector_data,
                                               *args, **kwargs)

            return vector_data

    @property
    def _dim(self):
        return self.shape[0]

    @property
    def _reconstruct_kwargs(self):
        return {}


class Matrix(np.ndarray):

    @classmethod
    def _preprocess_if_data_is_cls(cls, matrix, *args, **kwargs):
        # print "cls = ",cls
        # print "data = ",data
        # print "type(data) = ",type(data)
        # print "args = ",args
        # print "kwargs = ",kwargs
        # eval_cls = kwargs.get("eval_cls",1234)
        # object.__setattr__(data, "_eval_cls", eval_cls)
        pass

    @classmethod
    def _preprocess_before_init_return(cls, matrix, *args, **kwargs):
        pass

    def __new__(cls, data, dtype=None, force2d="as_row", *args, **kwargs):

        assert force2d in ("as_row", "as_col")

        if isinstance(data, cls):
            matrix_data = data
            cls._preprocess_if_data_is_cls(matrix=matrix_data, *args, **kwargs)
            return matrix_data
        else:

            if isinstance(data, (unicode, str)):
                # in case of matrix_data = np.array(unicode-type instance)
                # the shape of matrix_data would be () and len(matrix_data.shape) == 0
                matrix_data = np.array([data], dtype=dtype)
            else:
                matrix_data = np.array(data, dtype=dtype)

            if len(matrix_data.shape) > 2:
                matrix_data = np_tools.clean_no_data_tensors(matrix_data)

            if force2d == "as_row":
                if len(matrix_data.shape) < 2:
                    matrix_data = np.array([matrix_data])
            elif force2d == "as_col":
                if len(matrix_data.shape) < 2:
                    matrix_data = np.array([matrix_data]).T

            assert len(matrix_data.shape) == 2

            matrix_data = matrix_data.view(cls)
            cls._preprocess_before_init_return(matrix=matrix_data,
                                               *args, **kwargs)

            return matrix_data

    @property
    def _is_1d(self):
        return len(np_tools.clean_no_data_tensors(self).shape) == 1

    @property
    def _as_1d_array(self):
        assert self._is_1d
        return np_tools.clean_no_data_tensors(self)

    @property
    def _nrow(self):
        return self.shape[0]

    @property
    def _ncol(self):
        return self.shape[1]

    @property
    def _is_1d_row(self):
        return self.shape[0] == 1

    @property
    def _is_1d_col(self):
        return self.shape[1] == 1

    @property
    def _reconstruct_kwargs(self):
        _reconstr_kwargs = {}
        return _reconstr_kwargs

    @property
    def T(self):
        tr_array = np.array(self)
        return type(self)(data=tr_array.T, **self._reconstruct_kwargs)

    def _1d_ngram(self, n):
        assert self._is_1d
        ngram_results = list(tools.ngram(self.flatten(), n))
        return type(self)(data=ngram_results, **self._reconstruct_kwargs)

    def build_values_index(self):
        u, inv = np.unique(self, return_inverse=True)
        states_matrix = StatesMatrix(data=u, eval_cls=type(self))

        assert states_matrix._is_1d
        idx_matrix = IndexMatrix(data=inv).reshape(self.shape)
        return states_matrix, idx_matrix

    def build_row_struct_index(self):
        temp_self = self.copy()
        temp_self.dtype = np.dtype(zip(map(str, range(self._ncol)),
                                       [self.dtype] * self._ncol))

        u, i, inv = np.unique(temp_self, return_index=True,
                              return_inverse=True)

        idx_matrix = IndexMatrix(inv)

        states_matrix = StatesMatrix(self[i, :], eval_cls=type(u),
                                     is_row_struct=True)
        return states_matrix, idx_matrix

    def build_index_data_matrix(self, build_type="values"):
        assert build_type in ("values", "row_struct")
        if build_type == "values":
            return IndexedDataMatrix(*self.build_values_index())
        else:
            return IndexedDataMatrix(*self.build_row_struct_index())


class ValuesMatrix(Matrix):
    pass


class IndexMatrix(Matrix):
    pass


class IndexTransformVector(Vector):

    def __call__(self, trans_matrix, force2d="as_row"):
        return self.transform(trans_matrix=trans_matrix, force2d=force2d)

    def transform(self, trans_matrix, force2d="as_row"):
        _states_matrix_checker = isinstance(trans_matrix, StatesMatrix) and (trans_matrix._eval_cls == IndexMatrix)
        _index_matrix_checker = isinstance(trans_matrix, IndexMatrix)

        assert _states_matrix_checker or _index_matrix_checker
        assert self.validate_trans_matrix_value(trans_matrix)

        result_matrix = type(trans_matrix)(data=self[trans_matrix],
                                           force2d=force2d,
                                           **trans_matrix._reconstruct_kwargs)

        return result_matrix

    def validate_trans_matrix_value(self, trans_matrix):
        return trans_matrix.max().take(0) <= len(self)


class StatesMatrix(Matrix):

    @classmethod
    def _set_ref_data(cls, matrix, *args, **kwargs):
        ref_data = kwargs.get("ref_data", list())
        if len(ref_data) > 0:
            object.__setattr__(matrix, "_ref_data", ref_data)
        elif not ("_" in matrix.__dict__.keys()):
            object.__setattr__(matrix, "_ref_data", ref_data)

    @classmethod
    def _set_is_row_struct(cls, matrix, *args, **kwargs):
        is_row_struct = kwargs.get("is_row_struct", False)
        object.__setattr__(matrix, "_is_row_struct", is_row_struct)

    @classmethod
    def _set_eval_cls(cls, matrix, *args, **kwargs):
        eval_cls = kwargs.get("eval_cls", np.ndarray)
        object.__setattr__(matrix, "_eval_cls", eval_cls)

    @classmethod
    def _preprocess_if_data_is_cls(cls, matrix, *args, **kwargs):
        cls._set_eval_cls(matrix=matrix, *args, **kwargs)
        cls._set_is_row_struct(matrix=matrix, *args, **kwargs)
        cls._set_ref_data(matrix=matrix, *args, **kwargs)

    @classmethod
    def _preprocess_before_init_return(cls, matrix, *args, **kwargs):
        cls._set_eval_cls(matrix=matrix, *args, **kwargs)
        cls._set_is_row_struct(matrix=matrix, *args, **kwargs)
        cls._set_ref_data(matrix=matrix, *args, **kwargs)

    @property
    def _reconstruct_kwargs(self):
        _reconstr_kwargs = {}

        if self._has_ref_data:
            _reconstr_kwargs["ref_data"] = self._ref_data

        _reconstr_kwargs["eval_cls"] = self._eval_cls
        _reconstr_kwargs["is_row_struct"] = self._is_row_struct

        return _reconstr_kwargs

    @property
    def _has_ref_data(self):
        return len(self._ref_data) > 0

    def _eval_values_type(self, idx_mat):
        return self._as_1d_array[idx_mat].view(self._eval_cls).copy()

    def _eval_row_struct_type(self, idx_mat):
        return self[idx_mat._as_1d_array, :].view(self._eval_cls).copy()

    def _eval(self, idx_mat):

        assert isinstance(idx_mat, IndexMatrix)
        _row_struct_type_checkers = (idx_mat._is_1d and self._is_row_struct)
        _values_type_checkers = self._is_1d
        assert _values_type_checkers or _row_struct_type_checkers

        if _row_struct_type_checkers:
            return self._eval_row_struct_type(idx_mat=idx_mat)
        else:
            return self._eval_values_type(idx_mat=idx_mat)

    def add_ref_data(self, idx_data_matrix):
        assert isinstance(idx_data_matrix, IndexedDataMatrix)
        if not (idx_data_matrix in self._ref_data):
            self._ref_data.append(idx_data_matrix)

    def remove_ref_data(self, idx_data_matrix):
        assert isinstance(idx_data_matrix, IndexedDataMatrix)
        if idx_data_matrix in self._ref_data:
            self._ref_data.remove(idx_data_matrix)

    def pop_out_ref_data(self):
        while len(self._ref_data) > 0:
            yield self._ref_data.pop()

    def pop_out_clean_all_ref_data(self):
        return [xx for xx in self.pop_out_ref_data()]

    def create_new_ref_data_list(self):
        new_ref_data = list()
        self._ref_data = new_ref_data

    def update_all_ref_data(self, new_states_matrix, idx_trans_vec=None,
                            force2d="as_row"):

        for one_ref_data in self.pop_out_ref_data():
            one_ref_data.update_states_matrix(new_states_matrix=new_states_matrix,
                                              idx_trans_vec=idx_trans_vec,
                                              force2d=force2d)


class IndexedDataMatrix(object):
    def __init__(self, states_matrix, index_matrix):
        self.states_matrix = states_matrix
        self.index_matrix = index_matrix
        self.states_matrix.add_ref_data(self)

    @property
    def _nrow(self):
        return self.index_matrix._nrow

    @property
    def _ncol(self):
        return self.index_matrix._ncol

    @property
    def _data(self):
        return self.states_matrix._eval(self.index_matrix)

    def __repr__(self):
        return "{IndexedDataMatrix} " + self._data.__repr__()

    def update_states_matrix(self, new_states_matrix, idx_trans_vec=None,
                             force2d="as_row"):

        self.states_matrix = new_states_matrix
        self.states_matrix.add_ref_data(self)
        if isinstance(idx_trans_vec, IndexTransformVector):
            self.transform_index_matrix(idx_trans_vec=idx_trans_vec,
                                        force2d=force2d)

    def transform_index_matrix(self, idx_trans_vec, force2d="as_row"):
        self.index_matrix = idx_trans_vec(self.index_matrix, force2d=force2d)

    def _1d_ngram(self, n):
        self.index_matrix = self.index_matrix._1d_ngram(n)
        return self

    def update_index_matrix(self, new_index_matrix):
        # [BUG] self.states_matrix.remove_ref_data(self.index_matrix)
        # [BUG] self.states_matrix.add_ref_data(new_index_matrix)
        self.index_matrix = new_index_matrix

    def tile(self, reps=(1, 1), update=True):
        new_index_matrix = np.tile(self.index_matrix, reps)

        if update:
            self.update_index_matrix(new_index_matrix=new_index_matrix)
            return self
        else:
            return type(self)(states_matrix=self.states_matrix,
                              index_matrix=new_index_matrix)

    def extend_rows(self, n, update=True):
        return self.tile(reps=(n, 1), update=update)

    def extend_cols(self, n, update=True):
        return self.tile(reps=(1, n), update=update)


class MultiIndexedDataMatrix(np.ndarray):
    def __new__(cls, indexed_data_matrixes):

        # checking states_data_arrays are all StatesDataArray's instance
        assert check_type_of_args(IndexedDataMatrix, *indexed_data_matrixes)

        # checking indexed_data_matrixes have the same _nrow
        assert len(np.unique(np.array(map(lambda xx: xx._nrow,
                                          indexed_data_matrixes)))) == 1

        return np.array(indexed_data_matrixes).view(cls)

    @property
    def _ncols(self):
        return map(lambda xx: xx._ncol, self)

    @property
    def _nrows(self):
        return map(lambda xx: xx._nrow, self)

    @property
    def _nrow(self):
        assert len(np.unique(self._nrows)) == 1
        return self._nrows[0]

    @property
    def states_matrixes(self):
        return map(lambda xx: xx.states_matrix, self)

    @property
    def index_matrixes(self):
        return map(lambda xx: xx.index_matrix, self)



if __name__ == '__main__':
    pass
