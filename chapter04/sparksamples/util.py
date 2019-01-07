# -*- coding: utf-8 -*-
'''
Created on 2019年01月07日

@author: Zhukun Luo 
Jiangxi university of finance and economics
'''
import os
import sys
PATH = "/home/agnostic/Workspaces/MyEclipse CI/Spark-machine-learning/chapter04/data"
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
conf = SparkConf().setAppName("First Spark App").setMaster("local")

sc = SparkContext(conf=conf)
spark = SparkSession(sc)
from pyspark.sql import SQLContext
from pyspark.sql.types import  *
def get_user_data():
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
user_data=get_user_data()
print(user_data.collect())