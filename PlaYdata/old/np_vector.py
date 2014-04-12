'''
Created on Apr 12, 2014

@author: c3h3
'''

import numpy as np
from PlaYdata.util import np_tools
from PlaYdata.core.base import StatesMatrix, IndexMatrix


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


class IndexTransformVector(Vector):

    def __call__(self, trans_matrix, force2d="as_row"):
        return self.transform(trans_matrix=trans_matrix, force2d=force2d)

    def transform(self, trans_matrix, force2d="as_row"):
        _states_matrix_checker = isinstance(trans_matrix, StatesMatrix) and (trans_matrix._eval_cls == IndexMatrix)
        _index_matrix_checker = isinstance(trans_matrix, IndexMatrix)

        assert _states_matrix_checker or _index_matrix_checker
        assert self.validate_trans_matrix_value(trans_matrix)

        result_matrix = type(trans_matrix)(data=self[trans_matrix],
                                           force2d=force2d, **trans_matrix._reconstruct_kwargs)
        return result_matrix

    def validate_trans_matrix_value(self, trans_matrix):
        return trans_matrix.max().take(0) <= len(self)


if __name__ == '__main__':
    pass
