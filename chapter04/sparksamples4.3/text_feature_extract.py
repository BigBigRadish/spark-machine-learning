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
def extract_title(raw): #获取标题
    import re 
    grps = re.search("\((\w+)\)",raw) 
    if grps: 
        return raw[:grps.start()].strip() 
    else: return raw
raw_title_df=movie_df.select('moviename')
raw_title_df.show()
raw_title_df.createOrReplaceTempView("titles")        
spark.udf.register("extract_title", extract_title) 
title_df=spark.sql("select extract_title(moviename) as title from titles")
title_df.show()  
#分词和创建索引
title_rdd=title_df.rdd.map(lambda d: d.title.split(" "))#datafram转rdd
# print(title_rdd[5])
all_terms=title_df.rdd.flatMap(lambda d: d.title.split(" ")).distinct().collect()#flat操作
print(all_terms)
all_terms_dict={}
idx=0
for i in all_terms:
    all_terms_dict[i]=idx
    idx+=1
print(len(all_terms_dict))
print(all_terms_dict['Dead'])
print(all_terms_dict['Rooms'])
#创建标题的稀疏向量
def create_vector(terms, term_dict):
    from scipy import sparse as sp
    num_terms = len(term_dict)
    x = sp.lil_matrix((1,num_terms))
    for t in terms:
        if t in term_dict:
            idx = term_dict[t]
            x[0,idx] = 1
    return x
all_terms_bcast = sc.broadcast(all_terms_dict)
term_vectors = title_rdd.map(lambda terms: create_vector(terms,all_terms_bcast.value)).collect()#广播向量不适合在辞典较大时候使用
print(term_vectors[0])
    
    



