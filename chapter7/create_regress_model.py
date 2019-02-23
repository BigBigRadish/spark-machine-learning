# -*- coding: utf-8 -*-
'''
Created on 2019年02月23日

@author: Zhukun Luo 
Jiangxi university of finance and economics
'''
import os
import sys
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
conf = SparkConf().setAppName("Spark App").setMaster("local[4]")#默认分配线程

sc = SparkContext(conf=conf)
spark = SparkSession(sc)
raw_df = spark.read.csv('./data/hour.csv', header='true', inferSchema='true') #导入数据
print(raw_df.count())          #了解数据
raw_df.printSchema() #列出数据的详细属性信息
raw_df.select('cnt').show(10)
df1=raw_df.drop("instant","dteday","casual","registered")#丢弃列
df1.printSchema()
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col 
#全部转成double
df2=df1.withColumn("season", df1["season"].cast("double"))\
                                    .withColumn("yr", df1["yr"].cast("double"))\
                                    .withColumn("mnth", df1["mnth"].cast("double"))\
                                    .withColumn("holiday", df1["holiday"].cast("double"))\
                                    .withColumn("weekday", df1["weekday"].cast("double"))\
                                    .withColumn("workingday", df1["workingday"].cast("double"))\
                                    .withColumn("weathersit", df1["weathersit"].cast("double"))\
                                    .withColumn("cnt", df1["cnt"].cast("double"))\
                                    .withColumn("hr", df1["hr"].cast("double"))
df2.printSchema()
'''
封装成Pipline 数据类型转换与训练 (分类决策树)
'''
from pyspark.ml import Pipeline 
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler ,\
    VectorIndexer
from pyspark.ml.regression import DecisionTreeRegressor,LinearRegression

df3=df2.drop("cnt")
featureCols=df3.columns
vector_assembler = VectorAssembler(inputCols = featureCols,outputCol= 'rawfeatures')
vector_indexer=VectorIndexer(maxCategories=4, inputCol="rawfeatures", outputCol="features")
pipeline = Pipeline(stages=[vector_assembler , vector_indexer])
pipelineFit = pipeline.fit(df2)
dataset = pipelineFit.transform(df2)
dataset.show(20)
#将数据集分为训练集和测试集
train_df, test_df = dataset.randomSplit([0.7, 0.3])
train_df.show(20)
print (train_df.cache().count())
print (test_df.cache().count())
# 模型训练
lr = LinearRegression(featuresCol='features',labelCol='cnt',maxIter=5, regParam=0.0, solver="normal")
lrModel = lr.fit(train_df)
# 模型预测
prediction = lrModel.transform(test_df)
from pyspark.ml.evaluation import RegressionEvaluator
rmse=RegressionEvaluator(labelCol="cnt").evaluate(prediction)#评估
print(rmse)
'''
'''
