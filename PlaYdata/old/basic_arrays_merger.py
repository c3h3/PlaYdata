'''
Created on Apr 5, 2014

@author: c3h3
'''

import numpy as np
from PlaYdata.util.args_tools import check_type_of_args
from PlaYdata.core.basic_arrays import StatesDictionary, StatesArray
import PlaYdata.util.array_tools as tools


class StatesDictionaryMerger(list):
    def __init__(self, *states_dicts):
        
        # checking states_dicts are all StatesDictionary's instance
        assert check_type_of_args(StatesDictionary, *states_dicts)
        
        # checking states_dicts have the same dtype
        assert len(np.unique(np.array(map(lambda xx:xx.dtype.type,states_dicts)))) == 1
        self._dtype_type = states_dicts[0].dtype.type
        
        # checking states_dicts have the same _eval_cls
        assert len(np.unique(np.array(map(lambda xx:xx._eval_cls,states_dicts)))) == 1
        self._eval_cls = states_dicts[0]._eval_cls
        
        self._executed_merge = False
        
        list.__init__(self,states_dicts)
    
    
    @property
    def _unique_states_array_ids(self):
        return np.unique(np.array(map(lambda xx:id(xx._states_array),self)))
        
    def merge(self):
        if not self._executed_merge:
        
            if len(self._unique_states_array_ids) > 1:
                states_array_lens = map(lambda xx:len(xx._states_array),self)
                sector_position = map(lambda xx:slice(*xx),list(tools.ngram(np.cumsum([0] + states_array_lens),2)))
                join_all_states_arrays = np.concatenate(tuple(map(lambda xx:xx._states_array,self)),axis=0)
                u,i = np.unique(join_all_states_arrays,return_inverse=True)        
                ptrs_transforms = map(lambda xx:i[xx],sector_position)
        
                self._new_states_array = StatesArray(data=u, eval_cls=self._eval_cls)
                self._new_states_dict = StatesDictionary(states_array=self._new_states_array)
                self._ptrs_transforms = ptrs_transforms
        
            else:
                self._new_states_array = self[0]._states_array
                self._new_states_dict = StatesDictionary(states_array=self._new_states_array)
        
            self._executed_merge = True
    
        return self
    
    
    def update(self):
        
        if not self._executed_merge:
            self.merge()
        
        
        if len(self._unique_states_array_ids) > 1:
            for ptrs_trans, states_dict in zip(self._ptrs_transforms,self):
                states_dict.replace_states_dict(new_states_dict = self._new_states_dict, 
                                                ptrs_transform = ptrs_trans)
        
        else:
            for states_dict in self:
                states_dict.replace_states_dict(new_states_dict = self._new_states_dict)




if __name__ == '__main__':
    pass