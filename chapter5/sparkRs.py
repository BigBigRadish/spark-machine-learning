# -*- coding: utf-8 -*-
'''
Created on 2019年01月20日

@author: Zhukun Luo 
Jiangxi university of finance and economics
'''
#spark 构建推荐引擎
#提取有效特征
import os
import sys
from pyspark.mllib.linalg import DenseVector
from pyspark.mllib.recommendation import Rating, ALS
PATH = "/home/agnostic/Workspaces/MyEclipseCI/Spark-machine-learning/chapter05/data"
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
rawData = sc.textFile("./data//ml-100k/u.data")#导入数据
print(rawData.first())
rawRatings = rawData.map(lambda x: x.split('\t'))
print(rawRatings.take(5))

ratings = rawRatings.map(lambda x: Rating(int(x[0]),int(x[1]),float(x[2])))#提取特征
print(ratings.take(5))
#训练推荐模型
model = ALS.train(ratings, 50, 10, 0.01)
'''
rank:对应ALS模型中的因子个数,也就是在低阶近似矩阵中的隐含特征个数。因子个数一般越多越好。但它也会直接影响模型训练和保存时所需的内存开销,尤其是在用户和物品很多的时候。因此实践中该参数常作为训练效果与系统开销之间的调节参数。通常,其合理取值为10到200。
iterations:对应运行时的迭代次数。ALS能确保每次迭代都能降低评级矩阵的重建误差,但一般经少数次迭代后ALS模型便已能收敛为一个比较合理的好模型。这样,大部分情况下都没必要迭代太多次(10次左右一般就挺好)。
lambda:该参数控制模型的正则化过程,从而控制模型的过拟合情况。其值越高,正则化越严厉。该参数的赋值与实际数据的大小、特征和稀疏程度有关。和其他的机器学习模型一样,正则参数应该通过用非样本的测试数据进行交叉验证来调整。
作为示例,这里将使用的rank、iterations和lambda参数的值分别为20、10和0.01:
'''
userFeatures = model.userFeatures()
print (userFeatures.take(2))
print (model.userFeatures().count())
print (model.productFeatures().count())
'''
MLlib中标准的矩阵分解模型用于显式评级数据的处理。若要处理隐式数据,则可使用trainImplicit函数。其调用方式和标准的train模式类似,但多了一个可设置的alpha参数（也是一个正则化参数,lambda应通过测试和交叉验证法来设置）。
alpha参数指定了信心权重所应达到的基准线。该值越高则所训练出的模型越认为用户与他所没评级过的电影之间没有相关性。
'''
#模型预测
print(len(userFeatures.first()[1]))
predictRating = model.predict(789,123)
print (predictRating)
#topk
topKRecs = model.recommendProducts(789,10)
print('给用户userId推荐其喜欢的item：')
for rec in topKRecs:
    print (rec)
moviesForUser = ratings.groupBy(lambda x : x.user).mapValues(list).lookup(789)#789用户评级过的电影
print ('用户对%d部电影进行了评级'%len(moviesForUser[0]))
print ('源数据中用户(userId=789)喜欢的电影(item)：')
for i in sorted(moviesForUser[0],key=lambda x : x.rating,reverse=True): 
    print (i.product)
    
movies = sc.textFile("./data/ml-100k/u.item")
titles = movies.map(lambda line: (int(line.split('|')[0]),line.split('|')[1])).collectAsMap()
for i,rec in enumerate(topKRecs):#topk个电影的名字
    print ('rank:'+str(i)+' '+str(titles[rec.product])+':'+str(rec.rating))
    
#物品推荐
def cosineSImilarity(x,y):#余弦相似度
    return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
testx = np.array([1.0,2.0,3.0]) 
print cosineSImilarity(testx,testx)
itemId = 567
itemFactor = model.productFeatures().lookup(itemId)[0]
print(itemFactor)
sims = model.productFeatures().map(lambda (id,factor):(id,cosineSImilarity(np.array(factor), np.array(itemFactor)))) 
print(sims.sortBy(lambda (x,y):y,ascending=False).take(10))#取10个最相似的物品
#效果评估
#均方差
actual = moviesForUser[0][0] 
actualRating = actual.rating 
print('用户789对电影1012的实际评级',actualRating)
predictedRating = model.predict(789, actual.product)
print ('用户789对电影1012的预测评级',predictedRating)
squaredError = np.power(actualRating-predictedRating,2)
print('实际评级与预测评级的MSE',squaredError)

userProducts = ratings.map(lambda rating:(rating.user,rating.product)) 
print ('实际的评分:',userProducts.take(5)) 
#预测所有用户对电影的相应评分 
print (model.predictAll(userProducts).collect()[0])
predictions = model.predictAll(userProducts).map(lambda rating:((rating.user,rating.product), rating.rating)) 
print ('预测的评分:',predictions.take(5))
ratingsAndPredictions = ratings.map(lambda rating:((rating.user,rating.product),rating.rating)).join(predictions) 
print ('组合预测的评分和实际的评分:',ratingsAndPredictions.take(5)) 
MSE = ratingsAndPredictions.map(lambda ((x,y),(m,n)):np.power(m-n,2)).reduce(lambda x,y:x+y)/ratingsAndPredictions.count() 
print ('模型的均方误差:',MSE)
print ('模型的均方根误差:',np.sqrt(MSE))
#MAP K值平均准确率
def avgPrecisionK(actual, predicted, k): #MAP
    if len(predicted) > k: 
        predK = predicted[:k] 
    else: 
        predK = predicted 
        score = 0.0 
        numHits = 0.0 
        for i,p in enumerate(predK): 
            if p in actual and p not in predK: 
                numHits = numHits + 1 
                score = score + numHits/(i+1) 
        if not actual: 
            return 1.0 
        else: 
            return score/min(len(actual),k)
actualMovies = [rating.product for rating in moviesForUser[0]] 
predictMovies = [rating.product for rating in topKRecs] 
print ('实际的电影：',actualMovies)
print ('预测的电影：',predictMovies)
#计算map值
MAP10 = avgPrecisionK(actualMovies,predictMovies,10)
print(MAP10)
'''
全局MAPK的求解要计算对每一个用户的APK得分,再求其平均。
这就要为每一个用户都生成相应的推荐列表。
针对大规模数据处理时,这并不容易,但我们可以通过Spark将该计算分布式进行。
不过,这就会有一个限制,即每个工作节点都要有完整的物品因子矩阵。
这样它们才能独立地计算某个物品向量与其他所有物品向量之间的相关性。
然而当物品数量众多时,单个节点的内存可能保存不下这个矩阵。
'''
#首先,取回物品因子向量并用它来构建一个二维距征对象
itemFactors = model.productFeatures().map(lambda (id,factor):factor).collect()
itemMatrix = np.array(itemFactors) 
print(itemMatrix)
print(itemMatrix.shape)
#接下来,我们将该矩阵以一个广播变量的方式分发出去,以便每个工作节点都能访问到
imBroadcast = sc.broadcast(itemMatrix)
#现在可以计算每一个用户的推荐。
#这会对每一个用户因子进行一次map操作。
#在这个操作里,会对用户因子矩阵和电影因子矩阵做乘积,其结果为一个表示各个电影预计评级的向量(长度为1682,即电影的总数目)。
#之后,用预计评级对它们排序。
userVector = model.userFeatures().map(lambda (userId,array):(userId,np.array(array))) 
#print userVector.collect()[0][1].shape 
userVector = userVector.map(lambda (userId,x): (userId,imBroadcast.value.dot((np.array(x).transpose())))) 
#print userVector.collect()[0][1].shape 
userVectorId = userVector.map(lambda (userId,x) : (userId,[(xx,i) for i,xx in enumerate(x.tolist())])) 
#print userVectorId.collect()[0] 
sortUserVectorId = userVectorId.map(lambda (userId,x):(userId,sorted(x,key=lambda x:x[0],reverse=True))) 
sortUserVectorRecId = sortUserVectorId.map(lambda (userId,x): (userId,[xx[1] for xx in x])) 
#print sortUserVectorRecId.take(2)
sortUserVectorRecId.count()

userMovies = ratings.map(lambda rating: (rating.user,rating.product)).groupBy(lambda (x,y):x) 
print userMovies.take(3) 
userMovies = userMovies.map(lambda (userId,x):(userId, [xx[1] for xx in x] )) 
#print userMovies.take(1) 
#allAPK=sortUserVectorRecId.join(userMovies).map(lambda (userId,(predicted, actual)):(userId,avgPrecisionK(actual,predicted,10))) 
#print allAPK.map(lambda (x,y):y).reduce(lambda x,y:x+y)/sortUserVectorRecId.count() 
allAPK=sortUserVectorRecId.join(userMovies).map(lambda (userId,(predicted, actual)):avgPrecisionK(actual,predicted,2000)) 
# print allAPK.take(10) 
print allAPK.reduce(lambda x,y:x+y)/allAPK.count()
#mlib内置函数
from pyspark.mllib.evaluation import RegressionMetrics 
from pyspark.mllib.evaluation import RankingMetrics 
predictedAndTrue = ratingsAndPredictions.map(lambda ((userId,product),(predicted, actual)) :(predicted,actual)) 
print predictedAndTrue.take(5) 
regressionMetrics = RegressionMetrics(predictedAndTrue)
print("均方误差 = %f"%regressionMetrics.meanSquaredError)
print("均方根误差 = %f"% regressionMetrics.rootMeanSquaredError)
#MAP
sortedLabels = sortUserVectorRecId.join(userMovies).map(lambda (userId,(predicted, actual)):(predicted,actual)) 
# print sortedLabels.take(1) 
rankMetrics = RankingMetrics(sortedLabels) 
print("Mean Average Precision = %f" % rankMetrics.meanAveragePrecision)
print("Mean Average Precision(at K=10) = %f" % rankMetrics.precisionAt(10))





