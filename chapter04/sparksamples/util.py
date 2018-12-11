# -*- coding: utf-8 -*-
'''
Created on 2019年01月07日

@author: Zhukun Luo 
Jiangxi university of finance and economics
'''
import os
import sys
import numpy as np
PATH = "/home/agnostic/Workspaces/MyEclipse CI/Spark-machine-learning/chapter04/data"
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
conf = SparkConf().setAppName("First Spark App").setMaster("local")#默认分配线程

sc = SparkContext(conf=conf)
spark = SparkSession(sc)
from pyspark.sql import SQLContext
from pyspark.sql.types import  *
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
user_data=get_user_data()
print(user_data.collect())#及早求值
print(user_data.take(2))#取前俩个
num_users=user_data.count()#用户数量
print(num_users)
num_genders=len(user_data.groupBy('gender').count().collect())#计算性别数目
print(num_genders)
# num_occupations=user_data.map(lambda i: i[3]).distinct().count().sum()#不是记录集合不能使用
num_occupations=len(user_data.groupBy('occupation').count().collect())#计算职业的数目
print(num_occupations)
num_zipcodes=len(user_data.groupBy('zipCode').count().collect())#计算职业的数目
print(num_zipcodes)
import matplotlib
import matplotlib.pyplot as plt
def plot_user_age():#画出年龄分布的直方图
    user_data = get_user_data()
    user_ages = user_data.select('age').collect()
    user_ages_list = []#存值
    user_ages_len = len(user_ages)
    for i in range(0, (user_ages_len - 1)):
        user_ages_list.append(user_ages[i].age)
    plt.hist(user_ages_list, bins=20, color='lightblue', normed=True)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(16, 10)
    plt.show()
    plt.savefig('./user_age.png')
def plot_user_occupation():#统计职业分布
    user_data=get_user_data()
    user_occ=user_data.groupBy('occupation').count().collect() 
    print(user_occ)
    user_occ_len=len(user_occ)
    user_occ_list=[]
    for i  in range(0,(user_occ_len-1)):
        element=user_occ[i]
        count=element.__getattr__('count')
        tup=(element.occupation,count)
        user_occ_list.append(tup)
    x_axis1=np.array([c[0] for c in user_occ_list])
    y_axis1=np.array([c[1] for c in user_occ_list])
    x_axis=x_axis1[np.argsort(y_axis1)]
    y_axis=y_axis1[np.argsort(y_axis1)]
    pos=np.arange(len(x_axis))
    width=1.0
    ax=plt.axes()
    ax.set_xticks(pos+(width/2))
    ax.set_xticklabels(x_axis)
    plt.bar(pos,y_axis,width,color='lightblue')
    plt.xticks(rotation=45,fontsize='9')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.show()
    plt.savefig('./user_occupation.png')
        
if __name__ == "__main__":
#     plot_user_age()
    plot_user_occupation()