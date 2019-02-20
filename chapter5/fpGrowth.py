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
PATH = "/home/agnostic/Workspaces/MyEclipseCI/Spark-machine-learning/chapter04/data"
from pyspark.sql import SQLContext
from pyspark.sql.types import  *
from pyspark.sql import SparkSession
def get_rating_data():#创建dataframe
    custom_schema=StructType([
        StructField('user_id',IntegerType(),True),
        StructField('movie_id',IntegerType(),True),
        StructField('rating',IntegerType(),True),
        StructField('timestamp',IntegerType(),True)]
        )
    sql_context=SQLContext(sc)
    rating_df=sql_context.read.format('com.databricks.spark.csv') \
        .options(header='false', delimiter='\t') \
        .load("%s/ml-100k/u.data" % PATH, schema = custom_schema)#'\t'作为分割符
    return rating_df
rating_df=get_rating_data()#用户数据
print(rating_df.first())
rating_df.createOrReplaceTempView("df")#创建一个临时模板
rate_df=spark.sql("select int(user_id),int(movie_id),int(rating) from df")#转成int形
rate_df.show()
rate_data=rate_df.rdd.map(lambda d: d)#datafram转rdd
print(rate_data)
#load movie data and get movie name
def getMovieDataDf():#将电影数据转换为dataframe
    custom_schema = StructType([
    StructField("no", StringType(), True),
    StructField("moviename", StringType(), True),
    StructField("date", StringType(), True),
    StructField("f1", StringType(), True), StructField("url", StringType(), True),
    StructField("f2", IntegerType(), True), StructField("f3", IntegerType(), True),
    StructField("f4", IntegerType(), True), StructField("f5", IntegerType(), True),
    StructField("f6", IntegerType(), True), StructField("f7", IntegerType(), True),
    StructField("f8", IntegerType(), True), StructField("f9", IntegerType(), True),
    StructField("f10", IntegerType(), True), StructField("f11", IntegerType(), True),
    StructField("f12", IntegerType(), True), StructField("f13", IntegerType(), True),
    StructField("f14", IntegerType(), True), StructField("f15", IntegerType(), True),
    StructField("f16", IntegerType(), True), StructField("f17", IntegerType(), True),
    StructField("f18", IntegerType(), True), StructField("f19", IntegerType(), True)
    ])
    sql_context=SQLContext(sc)
    movie_df=sql_context.read.format('com.databricks.spark.csv') \
        .options(header='false', delimiter='|') \
        .load("%s/ml-100k/u.item" % PATH, schema = custom_schema)
    return movie_df
movie_df=getMovieDataDf()#电影数据
titles=movie_df.select('moviename')# get titles
print(titles.take(20))
z=[]
aj={}
i=0
for a in range(501,900):
    movieForUserX=rate_df.select('user_id').rdd.map(lambda d: d).lookup(a)
    movieForUserX_10=movieForUserX.sortBy(rating).take(10)
    movieForUserX_10_1=movieForUserX_10.map(lambda r: r.product)
    temp=""
    for i in movieForUserX_10_1:
        if(temp.__eq__("")):
            temp=str(i)
        else:
            temp=temp+" "+x
        aj[i]=temp
        i+=1
z=aj
transacation=z.map(_.split(" "))
rddx=sc.parallelize(transacation,2).cache()
fpg=FPGrowth()
model=fpg.train(rddx,minSupport=0.3,numPartitions=1)
FreqItemset=model.freqItemsets().collect()
for item in FreqItemset:
    print(item.items,item.freq)
    

