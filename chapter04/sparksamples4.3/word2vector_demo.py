# -*- coding: utf-8 -*-
'''
Created on 2019年1月18日 下午8:43:12
Zhukun Luo
Jiangxi University of Finance and Economics
'''
'''
使用gensim中的word2vector模块
'''
import gensim

def word2v_function():
    model = gensim.models.KeyedVectors.load_word2vec_format("wiki.en.text.vector", binary=False)
    model.most_similar("man")
    model.similarity("woman", "girl")
 
if __name__ == '__main__':
    word2v_function()