分类模型 适用的场景：
预测用户对online ads 的click possibility，binary class problem（yes/no）
检测欺诈，binary class prblem
拖欠贷款
picture ,video,voice class (multi class)
give a tag for news,web pages,other contents(multi class)
发现垃圾邮件，垃圾页面，网络入侵，其他恶意行为（binary and multi）
检查故障，比如计算机系统或者网络的故障检测
根据顾客或在用户购买产品或者使用服务的概率对他们进行排序
预测顾客或者用户中谁有可能停止使用某个 产品或服务
the tasks of this chapter：
discussion class model of MLlib
extract features from raw data with Spark
tain some model with MLlib
predict with model what has been tarin.
use some evaluation metrics for show model cability
increse model with chapter4's feature
check optimizer params with cross-validation
3 class models in Spark:linear model,Decision tree,Bayesion model
标准的线性模型使用对等函数，MLlib提供两种适合二分类的损失函数：Logistic loss and hinge loss（svm），0-1loss 不常用，因为不可微分，计算grad very difficult
与二分类logistic相比，多元logstic分归使用最大似然来计算分类概率
本章数据集来自Kaggle，https://www.kaggle.com/c/stumbleUpon/data
删除第一行： sed 1d ./train.tsv >./train_noheader.tsv

 