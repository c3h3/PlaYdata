'''
Created on Apr 5, 2014

@author: c3h3
'''


def ngram(text, n ,filter_list=[]):
    if isinstance(n, int):
        for k in range(len(text)-n+1):
            if not(text[k:k+n] in filter_list):
                yield text[k:k+n]
            
    if isinstance(n, list):
        for n_i in n:
            for xx in ngram(text=text, n=n_i, filter_list=filter_list):
                yield xx
                
                
def ngram_no_filter(text, n):
    if isinstance(n, int):
        for k in range(len(text)-n+1):
            yield text[k:k+n]
            
    if isinstance(n, list):
        for n_i in n:
            for xx in ngram(text=text, n=n_i):
                yield xx



if __name__ == '__main__':
    pass