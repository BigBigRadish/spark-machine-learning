# -*- coding: utf-8 -*-
'''
Created on 2019年02月26日

@author: Zhukun Luo 
Jiangxi university of finance and economics
'''
import os
import sys
import datetime 
from pyspark.mllib.linalg import DenseVector
PATH = "/home/agnostic/Workspaces/MyEclipseCI/Spark-machine-learning/chapter9/data"
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
import numpy as np
conf = SparkConf().setAppName("Spark App").setMaster("local[4]")#默认分配4个线程

sc = SparkContext(conf=conf)
#svd demo
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
 
rows = sc.parallelize([
    Vectors.sparse(5, {1: 1.0, 3: 7.0}),
    Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
    Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
])
mat = RowMatrix(rows)
# Compute the top 2 singular values and corresponding singular vectors.
svd = mat.computeSVD(2, computeU=True)
U = svd.U       # The U factor is a RowMatrix.
s = svd.s       # The singular values are stored in a local dense vector.
V = svd.V       # The V factor is a local dense matrix.
print(V)

 
mat = RowMatrix(rows)
# Compute the top 2 principal components.
# Principal components are stored in a local dense matrix.
pc = mat.computePrincipalComponents(2)
# Project the rows to the linear space spanned by the top 4 principal components.
projected = mat.multiply(pc)
print(projected.rows.collect())
'''
DenseMatrix([[-0.31278534,  0.31167136],
             [-0.02980145, -0.17133211],
             [-0.12207248,  0.15256471],
             [-0.71847899, -0.68096285],
             [-0.60841059,  0.62170723]])
[DenseVector([1.6486, -4.0133]), DenseVector([-4.6451, -1.1168]), DenseVector([-6.4289, -5.338])]
'''