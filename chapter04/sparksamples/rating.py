# -*- coding: utf-8 -*-
'''
Created on 2019年01月08日

@author: Zhukun Luo 
Jiangxi university of finance and economics
'''
#探索评级数据
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
def get_rating_data():
    return sc.textFile("%s/ml-100k/u.data" % PATH)
def get_movie_data():
    return sc.textFile("%s/ml-100k/u.item" % PATH)
def get_user_data():
    custom_schema = StructType([
        StructField("no", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("gender", StringType(), True),
        StructField("occupation", StringType(), True),
        StructField("zipCode", StringType(), True)
    ])
    from pyspark.sql import SQLContext
    from pyspark.sql.types import *

    sql_context = SQLContext(sc)

    user_df = sql_context.read \
        .format('com.databricks.spark.csv') \
        .options(header='false', delimiter='|') \
        .load("%s/ml-100k/u.user" % PATH, schema = custom_schema)
    return user_df
def explor_rating():
    rating_data_raw = get_rating_data()
    print (rating_data_raw.first())
    num_ratings = rating_data_raw.count()
    print ("Ratings: %d" % num_ratings)
    num_movies = get_movie_data().count()
    num_users = get_user_data().count()

    rating_data = rating_data_raw.map(lambda line: line.split("\t"))
    ratings = rating_data.map(lambda fields: int(fields[2]))
    max_rating = ratings.reduce(lambda x, y: max(x, y))
    min_rating = ratings.reduce(lambda x, y: min(x, y))
    mean_rating = ratings.reduce(lambda x, y: x + y) / float(num_ratings)
    median_rating = np.median(ratings.collect())
    ratings_per_user = num_ratings / num_users
    ratings_per_movie = num_ratings / num_movies
    print ("Min rating: %d" % min_rating)
    print ("Max rating: %d" % max_rating)
    print ("Average rating: %2.2f" % mean_rating)
    print ("Median rating: %d" % median_rating)
    print("Average # of ratings per user: %2.2f" % ratings_per_user)
    print("Average # of ratings per movie: %2.2f" % ratings_per_movie)
if __name__ == "__main__":
     explor_rating()   