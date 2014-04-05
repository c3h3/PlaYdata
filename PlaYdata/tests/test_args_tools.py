'''
Created on Apr 5, 2014

@author: c3h3
'''

from PlaYdata.util.args_tools import check_args, check_type_of_args


def test_check_args_1():
    assert check_args(lambda xx:isinstance(xx,int),*range(10)) == True
    
def test_check_args_2():
    assert check_args(lambda xx:isinstance(xx,int),*map(str,range(10))) == False


def test_check_type_of_args_1():
    test_args = range(10) + map(str,range(10)) + map(float,range(10)) + map(unicode,range(10))
    assert check_type_of_args([int,str,float,unicode], *test_args) == True

def test_check_type_of_args_2():
    test_args = range(10) + map(str,range(10)) + map(float,range(10)) + map(unicode,range(10))
    assert check_type_of_args([int,float,unicode], *test_args) == False




if __name__ == '__main__':
    pass