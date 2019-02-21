# -*- coding: utf-8 -*-
'''
Created on 2019年02月21日

@author: Zhukun Luo 
Jiangxi university of finance and economics
'''
#可视化数据集
import os
import sys
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
conf = SparkConf().setAppName("Spark App").setMaster("local[4]")#默认分配线程

sc = SparkContext(conf=conf)
spark = SparkSession(sc)
raw_df = spark.read.csv('./data/train.tsv', header='true', sep='\t', inferSchema='true') #导入数据
raw_df.count()           #了解数据
raw_df.printSchema() #列出数据的详细属性信息
raw_df.select('url', 'alchemy_category', 'alchemy_category_score', 'label').show(10)

from pyspark.sql.functions import col    # col():将一个字符串转换为DataFrame中列, 并获取此列的值
from  pyspark.sql.functions import udf 

def replace_question_func(x):#填充？为0
    return '0' if x == '?' else x
replace_question = udf(replace_question_func)

print(raw_df.columns[4:])#输出第4列之后的列名

# 将raw_df.columns[4:]数据转换成double类型并重命名与['url', 'alchemy_category']列连接 
df = raw_df.select(['url', 'alchemy_category'] + [ replace_question(col(column))\
                                                  .cast('double')\
                                                  .alias(column) for column in raw_df.columns[4:]]) 
df.printSchema()
df.select('url', 'alchemy_category', 'alchemy_category_score', 'label').show(10) 

#将数据集分为训练集和测试集
train_df, test_df = df.randomSplit([0.7, 0.3])

print (train_df.cache().count())
print (test_df.cache().count())
'''
1、alchemy_category 类别特征数据转换 
第一特征转换器、StringIndexer将文字的类别特征转换数字 
第二特征转换器、OneHotEncoder 将数值的 类别特征字段 转换为 多个字段的Vector 
2、VectorAssembler 特征的组合,第二特征转换器、将多个特征整合到一起
'''

'''
封装成Pipline 数据类型转换与训练 (分类决策树)
'''
from pyspark.ml import Pipeline 
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler 
from pyspark.ml.classification import DecisionTreeClassifier

'''
转换器和学习模型
参数说明： StringIndexer：inputCol -> 要转换的字段名称 outputCol -> 转换后的字段名称 
OneHotEncoder：可以将一个数值的类别特征字段转换为多个字段的Vector向量 
VectorAssembler: 指定合并哪些字段，输出的字段名称 
http://spark.apache.org/docs/2.2.0/ml-features.html#stringindexer 
http://spark.apache.org/docs/2.2.0/ml-features.html#onehotencoder 
http://spark.apache.org/docs/2.2.0/ml-features.html#vectorassembler 
http://spark.apache.org/docs/2.2.0/ml-classification-regression.html#decision-tree-classifier
'''

string_indexer = StringIndexer(inputCol = 'alchemy_category',outputCol = 'alchemy_category_index')
one_hot_encoder = OneHotEncoder(inputCol = 'alchemy_category_index', outputCol = 'alchemy_category_index_vector') 
assembler_inputs = ['alchemy_category_index_vector']+raw_df.columns[4:-1] #列连接
vector_assembler = VectorAssembler(inputCols = assembler_inputs,outputCol= 'features') 
dt = DecisionTreeClassifier(featuresCol='features',labelCol='label', impurity='gini',maxDepth=5,maxBins=32)#DecisionTreeClassifier 模型学习器 

#创建Pipline实例对象—训练模型与预测

#实例对象--按照数据处理顺序 
pipeline = Pipeline(stages=[string_indexer,one_hot_encoder,vector_assembler,dt]) #一个操作管道
pipleline_model = pipeline.fit(train_df) 
predict_df = pipleline_model.transform(test_df) 
predict_df.printSchema()

#PipelineModel 模型保存—加载—预测
#保存模型 
# help(pipleline_model.save) 
pipleline_model.save('./data/dtc-model') 
# 加载模型 
from pyspark.ml.pipeline import PipelineModel 
load_pipeline_model = PipelineModel.load('./data/dtc-model') 
# 预测 
load_pipeline_model.transform(test_df)\
.select('label', 'prediction', 'rawPrediction', 'probability')\
.show(20, truncate=False)
#使用TrainValidation获取最佳模型
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder 
help(TrainValidationSplit) 
# 构建一个 决策树分类算法 网格参数 #1.不纯度度量 #2.深度#3.最大分支数
param_grid = ParamGridBuilder()\
.addGrid(dt.impurity, ['gini', 'entropy'])\
.addGrid(dt.maxDepth, [5, 10, 20])\
.addGrid(dt.maxBins, [8, 16, 32])\
.build() 
print(type(param_grid))
for param in param_grid: 
    print param 
    # 针对二分类创建模型评估器 
from pyspark.ml.evaluation import BinaryClassificationEvaluator
binary_class_evaluator = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='rawPrediction') 
# 创建 TrainValidationSplit 实例对象 
"""
    __init__(self, estimator=None, estimatorParamMaps=None, 
             evaluator=None, trainRatio=0.75,  seed=None)
    参数解释：
        estimator：        模型学习器，针对哪个算法进行调整超参数
        estimatorParamMaps:算法调整的参数组合
        evaluator：        评估模型的评估器，比如二分类的话，使用auc面积
        trainRatio:        训练集与验证集 所占的比例，此处的值表示的是 训练集比例
""" 
train_validataion_split = TrainValidationSplit(estimator=dt, evaluator=binary_class_evaluator, estimatorParamMaps=param_grid, trainRatio=0.8) 
# 建立新的Pipeline实例对象，使用 train_validataion_split 代替 原先 dt 
tvs_pipeline = Pipeline(stages=[string_indexer, one_hot_encoder, vector_assembler, train_validataion_split]) 
# tvs_pipeline 进行数据处理、模型训练（找到最佳模型） 
tvs_pipeline_model = tvs_pipeline.fit(train_df) 
type(tvs_pipeline_model) 
best_model = tvs_pipeline_model.stages[3].bestModel 
#best_model # 评估 最佳模型 
predictions_df = tvs_pipeline_model.transform(test_df) 
model_auc = binary_class_evaluator.evaluate(predictions_df) 
print(model_auc)
'''
Cross-Validation
__init__(self, estimator=None, estimatorParamMaps=None, evaluator=None, numFolds=3, seed=None) 
假设 K-Fold的CrossValidation交叉验证 K = 3,将数据分为3个部分： 
1、A + B作为训练，C作为验证 2、B + C作为训练，A作为验证 3、A + C最为训练，
B作为验证 http://spark.apache.org/docs/2.2.0/ml-tuning.html#cross-validation
'''
# 导入模块 
from pyspark.ml.tuning import CrossValidator 
help(CrossValidator) 
# 构建 CrossValidator实例对象，设置相关参数 
cross_validator = CrossValidator(estimator=dt, evaluator=binary_class_evaluator, estimatorParamMaps=param_grid, numFolds=3) 
# 创建Pipeline 
cv_pipeline = Pipeline(stages=[string_indexer, one_hot_encoder, vector_assembler, cross_validator]) 
type(cv_pipeline) 
# 使用 cv_pipeline 进行训练与验证（交叉） 
cv_pipeline_model = cv_pipeline.fit(train_df) 
# 查看最佳模型 
best_model = cv_pipeline_model.stages[3].bestModel 
#best_model # 使用测试集评估最佳模型 
cv_predictions = cv_pipeline_model.transform(test_df) 
cv_model_auc = binary_class_evaluator.evaluate(cv_predictions) 
print (cv_model_auc)

'''
RandomForest Alogrithm

'''
# 导入随机森林分类算法模块 
from pyspark.ml.classification import RandomForestClassifier 
# 创建RFC实例对象 
rfc = RandomForestClassifier(labelCol='label', featuresCol='features', numTrees=10, featureSubsetStrategy="auto", maxDepth=5, maxBins=32, impurity="gini") 
help(RandomForestClassifier) #help获取模型的完整参数详情，以及模型的详情
# 创建Pipeline实例对象 
rfc_pipeline = Pipeline(stages=[string_indexer, one_hot_encoder, vector_assembler, rfc]) 
# 训练--预测--评估
rfc_pipeline_model = rfc_pipeline.fit(train_df) 
rfc_predictions = rfc_pipeline_model.transform(test_df) 
rfc_model_auc = binary_class_evaluator.evaluate(rfc_predictions) 
print(rfc_model_auc) #0.737350609213




