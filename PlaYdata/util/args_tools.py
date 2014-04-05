'''
Created on Apr 5, 2014

@author: c3h3
'''

import numpy as np
import inspect


def check_args(checker, *args):
    assert callable(checker)
    args_len = len(args)
    return np.ones(args_len,dtype=np.int)[np.array(map(checker,args))].sum() == args_len
    
    
def class_checker(xx):
    return inspect.isclass(xx)
    

def check_type_of_args(valid_type, *args):
    if isinstance(valid_type, (tuple,list)):
        assert check_args(class_checker,*valid_type)
        valid_types = tuple(valid_type)
    
    elif inspect.isclass(valid_type):
        valid_types = valid_type

    return check_args(lambda xx:isinstance(xx,valid_types),*args)    



if __name__ == '__main__':
    pass