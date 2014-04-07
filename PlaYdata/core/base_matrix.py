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
        # print "matrix = ",matrix
        # print "type(matrix) = ",type(matrix)
        # print "args = ",args
        # print "kwargs = ",kwargs
        # eval_cls = kwargs.get("eval_cls",1234)
        # object.__setattr__(matrix, "_eval_cls", eval_cls)
        pass

    @classmethod
    def _preprocess_before_init_return(cls, matrix, *args, **kwargs):
        pass

    def __new__(cls, data, dtype=None, force2d="as_row", *args, **kwargs):

        assert force2d in ("as_row", "as_col")
        if isinstance(data, cls):
            values_array = data
            cls._preprocess_if_data_is_cls(matrix=values_array,
                                           *args, **kwargs)
            return data

        else:

            values_array = np.array(data, dtype=dtype)

            if force2d == "as_row":
                if len(values_array.shape) < 2:
                    values_array = np.array([values_array])
            elif force2d == "as_col":
                if len(values_array.shape) < 2:
                    values_array = np.array([values_array]).T
            
            if len(values_array.shape) > 2:
                values_array = np_tools.clean_no_data_tensors(values_array)
            
            assert len(values_array.shape) == 2
            
            
            values_array = values_array.view(cls)
            cls._preprocess_before_init_return(matrix=values_array, *args, **kwargs)
            
            return values_array
    
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
    
    

class ValuesMatrix(Matrix):
    pass


class IndexMatrix(Matrix):
    pass


class StatesMatrix(Matrix):
    
    @classmethod
    def _set_eval_cls(cls, matrix, *args, **kwargs):
        eval_cls = kwargs.get("eval_cls", np.ndarray)
        object.__setattr__(matrix, "_eval_cls", eval_cls)
    
    
    @classmethod
    def _preprocess_if_data_is_cls(cls, matrix , *args, **kwargs):
        cls._set_eval_cls(matrix=matrix , *args, **kwargs)
    
    
    @classmethod
    def _preprocess_before_init_return(cls, matrix , *args, **kwargs):
        cls._set_eval_cls(matrix=matrix , *args, **kwargs)
        
    
    def _1d_eval(self, idx_mat):
        return self._as_1d_array[idx_mat].view(self._eval_cls).copy()

    


if __name__ == '__main__':
    pass
