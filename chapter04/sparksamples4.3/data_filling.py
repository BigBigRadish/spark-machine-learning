# -*- coding: utf-8 -*-
'''
Created on 2019年01月08日

@author: Zhukun Luo 
Jiangxi university of finance and economics
'''
#探索电影数据
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
PATH = "/home/agnostic/Workspaces/MyEclipse CI/Spark-machine-learning/chapter04/data"
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
def get_movie_data():
    return sc.textFile("%s/ml-100k/u.item" % PATH)
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
def replaceEmptyStr(v):#填充空值函数
    if v=='':
        return 1900
    else:
        return v
def convert_year(x):#年份转换函数
    try:
        return int(x[-4:])
    except:
        # there is a 'bad' data point with a blank year, which we set to 1900 and will filter out later
        return None    
movie_data=getMovieDataDf()
movie_data.createTempView("movie_data")
spark.udf.register("convert_year", convert_year)#需要定义一个日期转换函数
movie_years = spark.sql("select convert_year(date) as year from movie_data")#将日期数据转换为年份
movie_years.createTempView("movie_years")#datafram才能使用
years_filtered_valid=movie_years.filter("year!='1900'")
years_filtered_valid=years_filtered_valid.sort("year", ascending=True)#升序
years_filtered_valid.show()
print(movie_years.count())#查看总值的数目
print(movie_years.dropna().count())#查看非空值的数目
mean_vals=years_filtered_valid.select(mean(years_filtered_valid['year'])).collect()
mean_val = mean_vals[0][0] # to show the number
print(mean_val)
movie_years.na.fill(mean_val)
print(movie_years.filter("year==''").count())#空值填充




