'''
Created on Apr 6, 2014

@author: c3h3
'''

# Has bugs in case of
# > clean_no_data_tensors(np.array([[["12345"]]])
#
# def clean_no_data_tensors(np_array):
#    return np_array[map(lambda xx:0 if xx==1 else slice(None,None,None),np_array.shape)]

from PlaYdata.util.args_tools import check_args


def clean_no_data_tensors(np_array):

    clean_idx = map(lambda xx: 0 if xx == 1 else slice(None, None, None),
                    np_array.shape)
    if check_args(lambda xx: xx == 1, *np_array.shape):
        clean_idx[0] = slice(None, None, None)
    return np_array[clean_idx]


if __name__ == '__main__':
    pass
