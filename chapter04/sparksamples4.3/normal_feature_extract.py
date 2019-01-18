# -*- coding: utf-8 -*-
'''
Created on 2019年01月18日

@author: Zhukun Luo 
Jiangxi university of finance and economics
'''
'''
正则化特征提取
'''
import numpy as np
from pyspark.mllib.feature import Normalizer
import os
import sys
import datetime 
from pyspark.mllib.linalg import DenseVector
PATH = "/home/agnostic/Workspaces/MyEclipseCI/Spark-machine-learning/chapter04/data"
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
import numpy as np
conf = SparkConf().setAppName("Spark App").setMaster("local")#默认分配线程
sc = SparkContext(conf=conf)
spark = SparkSession(sc)
#使用Mlib正则化特征的实现
np.random.seed(42)
x = np.random.randn(10)
norm_x_2 = np.linalg.norm(x)
normalized_x = x / norm_x_2
print ("x:\n%s" % x)
print ("2-Norm of x: %2.4f" % norm_x_2)
print ("Normalized x:\n%s" % normalized_x)
print ("2-Norm of normalized_x: %2.4f" %np.linalg.norm(normalized_x))
#使用Mlib正则化特征
normalizer=Normalizer()
vector =sc.parallelize([x])
normalized_x_mlib=normalizer.transform(vector).first().toArray()
print("x:\n%s" % x)
print("2-Norm of x: %2.4f" % norm_x_2)
print("Normalized x:\n%s" % normalized_x)
print("Normalized x MLlib:\n%s" % normalized_x_mlib)
print("2-Norm of normalized_x_mllib: %2.4f" % np.linalg.norm(normalized_x_mlib))