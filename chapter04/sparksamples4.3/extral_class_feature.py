# -*- coding: utf-8 -*-
'''
Created on 2019年01月16日

@author: Zhukun Luo 
Jiangxi university of finance and economics
'''
#探索电影数据
import os
import sys
from pyspark.mllib.linalg import DenseVector
from pyspark.ml.feature import OneHotEncoder
PATH = "/home/agnostic/Workspaces/MyEclipseCI/Spark-machine-learning/chapter04/data"
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
conf = SparkConf().setAppName("Spark App").setMaster("local")#默认分配线程

sc = SparkContext(conf=conf)
spark = SparkSession(sc)
from pyspark.sql import SQLContext
from pyspark.sql.types import  *

from pyspark.sql.functions import mean
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession
def get_occupation_data():
    return sc.textFile("%s/ml-100k/u.occupation" % PATH)
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
def get_user_data():#创建dataframe
    custom_schema=StructType([
        StructField('no',StringType(),True),
        StructField('age',IntegerType(),True),
        StructField('gender',StringType(),True),
        StructField('occupation',StringType(),True),
        StructField('zipCode',StringType(),True)]
        )
    sql_context=SQLContext(sc)
    user_df=sql_context.read.format('com.databricks.spark.csv') \
        .options(header='false', delimiter='|') \
        .load("%s/ml-100k/u.user" % PATH, schema = custom_schema)
    return user_df
rating_df=get_rating_data()#评级数据
movie_df=getMovieDataDf()#电影数据
user_df=get_user_data()#用户数据
# print(movie_df.select('url'))
rating_group=rating_df.groupBy('rating')
rating_group.count().show()
rating_byuser_local=rating_df.groupBy('user_id').count()
count_rating_byuser_local=rating_byuser_local.count()
rating_byuser_local.show(len(rating_byuser_local.collect()))
occupation_df=user_df.select('occupation').distinct()
occupation_df.sort('occupation').show()
occupation_df_collect=occupation_df.collect()
print(occupation_df_collect)
cols=occupation_df.columns
all_occupation_dict={}
j=0
for i in occupation_df_collect:#这是一个二维数组
    print(i[0])
    all_occupation_dict[str(i[0])]=str(j)
    j+=1
print(all_occupation_dict['doctor'])
k=len(all_occupation_dict)
binary_x=DenseVector(0)
print(binary_x)



