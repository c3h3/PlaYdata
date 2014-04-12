'''
Created on Apr 7, 2014

@author: c3h3
'''

import numpy as np
from PlaYdata.core.base_matrix import Matrix
import PlaYdata.util.np_tools as np_tools

class Vector(np.ndarray):
    _convert_to_matrix_cls = Matrix
    
    @classmethod
    def _preprocess_if_data_is_cls(cls, vector,*args, **kwargs):
        pass
        
    @classmethod
    def _preprocess_before_init_return(cls, vector ,*args, **kwargs):
        pass
    
    def __new__(cls, data, dtype=None, *args, **kwargs):
        
        if isinstance(data, cls):
            values_vector = data
            cls._preprocess_if_data_is_cls(vector=values_vector,*args, **kwargs)
            return values_vector
        
        else:
            vector_data = np.array(data, dtype=dtype)
            
            if len(vector_data.shape) > 1:
                vector_data = np_tools.clean_no_data_tensors(vector_data)
            
            assert len(vector_data.shape) == 1
            
            vector_data = vector_data.view(cls)
            cls._preprocess_before_init_return(vector=vector_data,*args, **kwargs)
            
            return vector_data
        
    @property
    def _dim(self):
        return self.shape[0]
    

class ValuesVector(Vector):
    pass
    

class IndexVector(Vector):
    pass


class StatesVector(Vector):
    pass

if __name__ == '__main__':
    pass