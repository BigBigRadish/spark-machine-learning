# -*- coding: utf-8 -*-
'''
Created on 2019年03月02日

@author: Zhukun Luo 
Jiangxi university of finance and economics
'''
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
PATH = "/home/agnostic/Workspaces/MyEclipse CI/Spark-machine-learning/chapter04/data"
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
conf = SparkConf().setAppName("Spark App").setMaster("local[4]")#默认分配线程

sc = SparkContext(conf=conf)
spark = SparkSession(sc)
from pyspark.ml.feature import Word2Vec
documentDF = spark.createDataFrame([
        ("Hi I heard about Spark".split(" "), ),
        ("I wish Java could use case classes".split(" "), ),
        ("Logistic regression models are neat".split(" "), )
    ], ["text"])
documentDF.show(20)
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")#设置特征向量为3#这是句向量
model = word2Vec.fit(documentDF)

result = model.transform(documentDF)
for row in result.collect():
    text, vector = row
    print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))
