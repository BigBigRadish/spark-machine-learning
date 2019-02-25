# -*- coding: utf-8 -*-
'''
Created on 2019年02月25日

@author: Zhukun Luo 
Jiangxi university of finance and economics
'''
import os
import sys
import datetime 
from pyspark.mllib.linalg import DenseVector
PATH = "/home/agnostic/Workspaces/MyEclipseCI/Spark-machine-learning/chapter8/data"
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
import numpy as np
conf = SparkConf().setAppName("Spark App").setMaster("local[4]")#默认分配4个线程

sc = SparkContext(conf=conf)
spark = SparkSession(sc)
from pyspark.sql import SQLContext
from pyspark.sql.types import  *

from pyspark.sql.functions import mean
from pyspark.sql.functions import udf
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
rating_df=get_rating_data()
rating_df.printSchema()
rating_df=rating_df.drop("timestamp")
rating_df.show(10)
trainset,testset=rating_df.randomSplit([0.8,0.2])#随机划分数据集
from pyspark.mllib.recommendation import  ALS
#训练推荐模型
model = ALS.train(rating_df, 50, 10, 0.01)#50个factors,10 iterations,0.01 iter rate
userFeatures = model.userFeatures()#user features
print(userFeatures.take(10))
movieFeatures = model.productFeatures()#product features
#K-mean训练模型
user_vectors = userFeatures.map(lambda (id,vec):vec)#转换成向量
movie_vectors = movieFeatures.map(lambda (id,vec):vec)#转换成向量
#归一化
from pyspark.mllib.linalg.distributed import RowMatrix
moive_matrix = RowMatrix(movie_vectors)#将vector转换成矩阵的形式
user_matrix = RowMatrix(user_vectors)

from pyspark.mllib.stat import MultivariateStatisticalSummary#计算vector的统计库
desc_moive_matrix = MultivariateStatisticalSummary(moive_matrix.rows) 
desc_user_matrix = MultivariateStatisticalSummary(user_matrix.rows) 
print('Movie factors mean:',desc_moive_matrix.mean())
print('Movie factors variance:',desc_user_matrix.mean())
print('User factors mean:',desc_moive_matrix.variance()) 
print('User factors variance:',desc_user_matrix.variance())

#训练聚类模型
from pyspark.mllib.clustering import KMeans 
num_clusters = 5 
num_iterations = 20 
# num_runs =3 
movie_cluster_model = KMeans.train(movie_vectors,num_clusters, num_iterations) 
movie_cluster_model_coverged = KMeans.train(movie_vectors,num_clusters,100) 
user_cluster_model = KMeans.train(user_vectors,num_clusters,num_iterations) 
predictions = movie_cluster_model.predict(movie_vectors) 
print('对前十个样本的预测标签为:'+",".join([str(i) for i in predictions.take(10)]))

#在MovieLens数据集计算性能
movie_cost = movie_cluster_model.computeCost(movie_vectors) 
user_cost = user_cluster_model.computeCost(user_vectors) 
print("WCSS for movies: %f"%movie_cost)
print("WCSS for users: %f"%user_cost)
'''
WCSS for movies: 10475.605245
WCSS for users: 4291.459190
'''
#聚类模型参数调优
'''
，K-均值模型只有一个可以调的参数，就是K，即类中心数目。通过交叉验证选择K
类似分类和回归模型，我们可以应用交叉验证来选择模型最优的类中心数目。这和监督学习的过程一样。
需要将数据集分割为训练集和测试集，然后在训练集上训练模型，在测试集上评估感兴趣的指标的性能。
如下代码用60/40划分得到训练集和测试集，并使用MLlib内置的WCSS类方法评估聚类模型的性能
'''
train_test_split_movies = movie_vectors.randomSplit([0.6,0.4],123) 
train_movies = train_test_split_movies[0] 
test_movies = train_test_split_movies[1] 
for k in [2,3,4,5,10,20,25]: 
    k_model = KMeans.train(train_movies, num_iterations, k) 
    cost = k_model.computeCost(test_movies) 
    print('WCSS for k=%d : %f'%(k,cost))
'''
WCSS for movies: 10598.460377
WCSS for users: 4277.108029
WCSS for k=2 : 3918.444838
WCSS for k=3 : 3845.078982
WCSS for k=4 : 3826.484943
WCSS for k=5 : 3880.857057
WCSS for k=10 : 3834.649329
WCSS for k=20 : 3842.751645
WCSS for k=25 : 3838.462133
'''
#用户聚类下的k值调优
train_test_split_movies = user_vectors.randomSplit([0.6,0.4],123) 
train_users = train_test_split_movies[0] 
test_users = train_test_split_movies[1] 
for k in [2,3,4,5,10,20,25,28,30]: 
    k_model = KMeans.train(train_users,num_iterations,k) 
    cost = k_model.computeCost(test_users) 
    print('WCSS for k=%d : %f'%(k,cost))






