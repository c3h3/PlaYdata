'''
Created on Apr 5, 2014

@author: c3h3
'''

import numpy as np
import PlaYnlp.tokenizer as tkr 


class ValuesArray(np.ndarray):
    def __new__(cls, data, dtype=None):
        if isinstance(data, cls):
            return data
        else:
            values_array = np.array(data, dtype=dtype).view(cls)
            return values_array
        
    def decompose_into_states_ptrs(self):
        u,i = np.unique(self, return_inverse=True)
        states_array = StatesArray(data=u, eval_cls=type(self))
        ptr_array = PtrArray(data=i)
        return states_array, ptr_array
    
    def to_states_data_array(self):
        return StatesDataArray(*self.decompose_into_states_ptrs())
    
    
class StatesArray(np.ndarray):
    def __new__(cls, data, dtype=None, eval_cls=np.ndarray):
        if isinstance(data, cls):
            data._eval_cls = eval_cls
            return data
        else:
            states_array = np.unique(np.array(data, dtype=dtype)).view(cls)
            states_array._eval_cls = eval_cls
            return states_array
    
    def _eval(self, ptr_array):
        assert isinstance(ptr_array, PtrArray)
        return self[ptr_array].view(self._eval_cls).copy()
        
    

class PtrArray(np.ndarray):
    def __new__(cls, data, dtype=None):
        if isinstance(data, cls):
            return data
        else:
            data_array = np.array(data, dtype).view(cls)
            return data_array
    
    def transform(self, ptrs_transform):
        return type(self)(data = ptrs_transform[self])

    def ngram(self, n):
        ngram_results = list(tkr.ngram(self,n))
        return type(self)(data=ngram_results, eval_cls=self._eval_cls)
    
    @property
    def T(self):
        new_np_data = np.array(self).T
        return type(self)(data=new_np_data)
        
    def decompose_into_states_ptrs(self):
        u,i = np.unique(self, return_inverse=True)
        states_array = StatesArray(data=u, eval_cls=type(self))
        ptr_array = PtrArray(data=i)
        return states_array, ptr_array

    
    @property
    def _dtype(self):
        return self._states_array.dtype
    
    
    
class StatesDictionary(object):
    def __init__(self, states_array):
        assert isinstance(states_array, StatesArray)
        self._states_array = states_array
        self._referred_by = []
        
        
    def __repr__(self):
        return u"{StatesDictionary} " + self._states_array.__repr__()
    
    def eval_ptr_array(self, ptr_array):
        assert isinstance(ptr_array, PtrArray)
        return self._states_array._eval(ptr_array).copy()
    
    
    def update_states_array(self, new_states_array, ptrs_transform=None):
        self._states_array = new_states_array
        for one_states_data_array in self._referred_by:
            if ptrs_transform != None:
                one_states_data_array._ptr_array = one_states_data_array._ptr_array.transform(ptrs_transform)
    
    def replace_states_dict(self, new_states_dict, ptrs_transform=None):
        
        assert isinstance(new_states_dict,type(self))
        
        for one_states_data_array in self._referred_by:
            self._referred_by.remove(one_states_data_array)
            new_states_dict._referred_by.append(one_states_data_array)
            one_states_data_array._states_dict = new_states_dict
            if ptrs_transform != None:
                one_states_data_array._ptr_array = one_states_data_array._ptr_array.transform(ptrs_transform)
    
    
    @property
    def dtype(self):
        return self._states_array.dtype
    
    @property
    def _eval_cls(self):
        return self._states_array._eval_cls
    
    
    
    
class StatesDataArray(object):
        
    def __init__(self, states_array, ptr_array,
                 _ptr_array_cls = PtrArray,
                 _ptr_array_eval_cls = ValuesArray):
        
        self._ptr_array_cls = _ptr_array_cls
        self._ptr_array_eval_cls = _ptr_array_eval_cls
        
        assert isinstance(states_array,(StatesDictionary,StatesArray))
        assert isinstance(ptr_array, self._ptr_array_cls)
        
        self._ptr_array = ptr_array    
        
        if isinstance(states_array,StatesDictionary):
            self._states_dict = states_array
        else:
            self._states_dict = StatesDictionary(states_array=states_array)
        
        self._states_dict._referred_by.append(self)
    
    
    def __repr__(self):
        return u"{StatesDataArray} " + self._data.__repr__()
    
    
    @property
    def _data(self):
        return self._states_dict.eval_ptr_array(self._ptr_array)




if __name__ == '__main__':
    pass