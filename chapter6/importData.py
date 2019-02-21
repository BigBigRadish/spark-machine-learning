# -*- coding: utf-8 -*-
'''
Created on 2019年02月21日

@author: Zhukun Luo 
Jiangxi university of finance and economics
'''
#spark导入数据
import os
import sys
PATH = "/home/agnostic/Workspaces/MyEclipseCI/Spark-machine-learning/chapter05/data"
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
conf = SparkConf().setAppName("Spark App").setMaster("local[4]")#默认分配线程

sc = SparkContext(conf=conf)
spark = SparkSession(sc)
aw_df = spark.read.csv('./data/train.tsv', header='true', sep='\t', inferSchema='true')
aw_df.show(20)