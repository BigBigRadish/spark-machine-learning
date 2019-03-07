# -*- coding: utf-8 -*-
'''
Created on 2019年03月02日

@author: Zhukun Luo 
Jiangxi university of finance and economics
'''
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
conf = SparkConf().setAppName("Spark App").setMaster("local[4]")#默认分配线程

sc = SparkContext(conf=conf)
spark = SparkSession(sc)
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF,Tokenizer
#准备测试数据
# Prepare training documents from a list of (id, text, label) tuples.
training = spark.createDataFrame([
(0L,"a b c d e spark",1.0),
(1L,"b d",0.0),
(2L,"spark f g h",1.0),
(3L,"hadoop mapreduce",0.0)],["id","text","label"])
#构建机器学习流水线
# Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
tokenizer =Tokenizer(inputCol="text", outputCol="words")
hashingTF =HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr =LogisticRegression(maxIter=10, regParam=0.01)
pipeline =Pipeline(stages=[tokenizer, hashingTF, lr])
#训练出model
# Fit the pipeline to training documents.
model = pipeline.fit(training)
#测试数据
# Prepare test documents, which are unlabeled (id, text) tuples.
test = spark.createDataFrame([
(4L,"spark i j k"),
(5L,"l m n"),
(6L,"mapreduce spark"),
(7L,"apache hadoop")],["id","text"])
#预测，打印出想要的结果
# Make predictions on test documents and print columns of interest.
prediction = model.transform(test)
selected = prediction.select("id","text","prediction")
for row in selected.collect():
    print(row)