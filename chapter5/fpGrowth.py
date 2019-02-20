# -*- coding: utf-8 -*-
'''
Created on 2019年02月20日

@author: Zhukun Luo 
Jiangxi university of finance and economics
'''
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.mllib.fpm import FPGrowth
import numpy as np
conf = SparkConf().setAppName("Spark App").setMaster("local[2]")#默认分配线程

sc = SparkContext(conf=conf)
spark = SparkSession(sc)
from pyspark.sql import SQLContext
from pyspark.sql.types import  *
#fpgworth algorithm
#创建一个多为数组
def sp(l):
    return l.split(' ')
transacation = ["r z h k p","z y x w v u t s","x o n r","x z y m t s q e",\
                "z","x z y r q t p"]
transacation_1=map(sp, transacation)
print(transacation_1)
rdd=sc.parallelize(transacation_1, 2).cache()
fpg=FPGrowth()
model=fpg.train(rdd,minSupport=0.3,numPartitions=1)
#minsupport：表示一个物体为高频物品所需的最小概率，10 此交易中出现三次，其对应的support值为3/10=0.3
#numPartitions:分区数目，以并行工作
FreqItemset=model.freqItemsets().collect()
for item in FreqItemset:
    print(item.items,item.freq)
'''
([u't', u'x'], 3)
([u't', u'x', u'z'], 3)
([u't', u'z'], 3)
([u's'], 2)
([u's', u't'], 2)
([u's', u't', u'x'], 2)
([u's', u't', u'x', u'z'], 2)
([u's', u't', u'z'], 2)
([u's', u'x'], 2)
([u's', u'x', u'z'], 2)
'''


