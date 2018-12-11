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

from pyspark.sql.functions import udf
from pyspark.sql import SparkSession
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
def convert_year(x):#年份转换函数
    try:
        return int(x[-4:])
    except:
        # there is a 'bad' data point with a blank year, which we set to 1900 and will filter out later
        return 1900
def getMovieYearsCountSorted():#对电影年份进行排序
#     movie_data_df=getMovieDataDf()
    movie_data=sc.textFile("%s/ml-100k/u.item" % PATH)
#     movie_years = movie_data_df.select('date').first()
#     print(movie_years)
#     movie_data_df.createTempView("movie_data")
#     spark.udf.register("convert_year", convert_year)#需要定义一个日期转换函数
#     movie_years = spark.sql("select convert_year(date) as year from movie_data")#将日期数据转换为年份
#     print(movie_years.collect())
    movie_fields = movie_data.map(lambda lines: lines.split("|"))
    years=movie_fields.map(lambda fields: fields[2]).map(lambda x: convert_year(x))
    years_filtered = years.filter(lambda x: x != '1900')
    movie_years_count=years.groupBy('year').count()
    print(len(movie_fields.first()))
    movie_ages = years_filtered.map(lambda yr: 1998-yr).countByValue()
    values = movie_ages.values()
    print(values)
    bins = movie_ages.keys()
#     print(bins)
#     plt.hist(values, bins=bins, color='lightblue', density=True)
#     fig = plt.gcf()
#     fig.set_size_inches(16,10)
#     plt.savefig('./movie_age.jpg')
#     plt.show()
#     print(bins)
#     plt.hist2d(movie_years_count.keys(), movie_years_count.values(), color='lightblue', density=True)
#     fig = plt.gcf()
#     fig.set_size_inches(16,10)
#     plt.savefig('./movie_age_1.jpg')
#     plt.show()
if __name__ == "__main__":
#     user_item=getMovieDataDf()
#     print(user_item.take(2))
    getMovieYearsCountSorted()
