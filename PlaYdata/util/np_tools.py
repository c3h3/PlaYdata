'''
Created on Apr 6, 2014

@author: c3h3
'''


def clean_no_data_tensors(np_array):
    return np_array[map(lambda xx:0 if xx==1 else slice(None,None,None),np_array.shape)]




if __name__ == '__main__':
    pass