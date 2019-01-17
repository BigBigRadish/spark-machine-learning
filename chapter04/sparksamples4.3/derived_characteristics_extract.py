# -*- coding: utf-8 -*-
'''
Created on 2019年01月17日

@author: Zhukun Luo 
Jiangxi university of finance and economics
'''
#探索电影数据
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
'''
首先使用map将时间戳属性转换为Pythonint类型。然后通过extract_datetime函数将各时间戳转为datetime类型的对象,进而提取出其小时数
'''
def str2float(s):#str to float
    def fn(x,y):
        return x*10+y
    n=s.index('.')
    s1=list(map(int,[x for x in s[:n]]))
    s2=list(map(int,[x for x in s[n+1:]]))
    return reduce(fn,s1)+reduce(fn,s2)/(10**len(s2))#乘幂
# print('\'123.456\'=',str2float('123.456'))
def extract_datetime(ts): #得到小时数目
    
    return datetime.datetime.fromtimestamp(ts).hour
rating_df.createOrReplaceTempView("df")#创建一个
spark.udf.register("extract_datetime", extract_datetime) 
timestamps_df=spark.sql("select extract_datetime(timestamp) as hour from df")
timestamps_df.show()
# timestamps = rating_df.select('timestamp')
# hour_of_day = timestamps.map(lambda ts: extract_datetime(ts).hour) 
# hour_of_day.take(5)
#将点钟数转换为时间段
def assignTod(hour):
    if(hour>=7 and hour<12):
        return "morning"
    elif(hour>=12 and hour<14):
        return "lunch"
    elif(hour>=14 and hour<18):
        return "afternoon"
    elif(hour>=18 and hour<23):
        return "evening"
    elif(hour>=23 and hour<=24):
        return "night"
    elif(hour<7):
        return "night"
    else:
        return "error"
timestamps_df.createOrReplaceTempView("time_df")        
spark.udf.register("assignTod", assignTod) 
tod_df=spark.sql("select assignTod(hour) as tod from time_df")
tod_df.show()       
        
