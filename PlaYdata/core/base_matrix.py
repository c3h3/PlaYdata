'''
Created on Apr 7, 2014

@author: c3h3
'''

import numpy as np
import PlaYdata.util.np_tools as np_tools
import PlaYdata.util.tools as tools


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

            matrix_data = np.array(data, dtype=dtype)

            if force2d == "as_row":
                if len(matrix_data.shape) < 2:
                    matrix_data = np.array([matrix_data])
            elif force2d == "as_col":
                if len(matrix_data.shape) < 2:
                    matrix_data = np.array([matrix_data]).T

            if len(matrix_data.shape) > 2:
                matrix_data = np_tools.clean_no_data_tensors(matrix_data)

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

    def _1d_ngram(self, n):
        assert self._is_1d
        ngram_results = list(tools.ngram(self.flatten(), n))
        return type(self)(data=ngram_results)

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


class ValuesMatrix(Matrix):
    pass


class IndexMatrix(Matrix):
    pass


class StatesMatrix(Matrix):

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

    @classmethod
    def _preprocess_before_init_return(cls, matrix, *args, **kwargs):
        cls._set_eval_cls(matrix=matrix, *args, **kwargs)
        cls._set_is_row_struct(matrix=matrix, *args, **kwargs)

    def _eval_1d_self(self, idx_mat):
        return self._as_1d_array[idx_mat].view(self._eval_cls).copy()

    def _eval_1d_idx_mat(self, idx_mat):
        return self[idx_mat._as_1d_array, :].view(self._eval_cls).copy()

    def _eval(self, idx_mat):

        assert isinstance(idx_mat, IndexMatrix)
        assert self._is_1d or (idx_mat._is_1d and self._is_row_struct)

        if idx_mat._is_1d and self._is_row_struct:
            return self._eval_1d_idx_mat(idx_mat=idx_mat)
        else:
            return self._eval_1d_self(idx_mat=idx_mat)


if __name__ == '__main__':
    pass
