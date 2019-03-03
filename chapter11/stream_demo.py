# -*- coding: utf-8 -*-
'''
Created on 2019年03月03日

@author: Zhukun Luo 
Jiangxi university of finance and economics
'''
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
# 创建本地的SparkContext对象，包含4个执行线程
sc = SparkContext("local[4]", "streamwordcount")
# 创建本地的StreamingContext对象，处理的时间片间隔时间，设置为3s
ssc = StreamingContext(sc, 3)

'''
创建DStream对象
我们需要连接一个打开的 TCP 服务端口，从而获取流数据，这里使用的源是TCP Socket，所以使用socketTextStream()函数：
'''
# 创建DStream，指明数据源为socket：来自localhost本机的8888端口
lines = ssc.socketTextStream("localhost", 8888)

'''
对DStream进行操作
我们开始对lines进行处理，首先对当前2秒内获取的数据进行分割并执行标准的MapReduce流程计算。
'''
# 使用flatMap和Split对3秒内收到的字符串进行分割
words = lines.flatMap(lambda line: line.split(" "))
# map操作将独立的单词映射到(word，1)元组
pairs = words.map(lambda word: (word, 1))
# reduceByKey操作对pairs执行reduce操作获得（单词，词频）元组
wordCounts = pairs.reduceByKey(lambda x, y: x + y)

'''
输出数据
'''
# 输出文件夹的前缀，Spark Streaming会自动使用当前时间戳来生成不同的文件夹名称
outputFile = "./output/"
# 将结果输出
wordCounts.saveAsTextFiles(outputFile)
'''
启动应用
要使程序在Spark Streaming上运行起来，需要执行Spark Streaming启动的流程，调用start()函数启动，awaitTermination()函数等待处理结束的信号。
'''
# 启动Spark Streaming应用
ssc.start() 
ssc.awaitTermination()

#打开终端：nc -lk 8888
'''
Netcat 或者叫 nc 是 Linux 下的一个用于调试和检查网络工具包。可用于创建 TCP/IP 连接，最大的用途就是用来处理 TCP/UDP 套接字。
这里我们将通过一些实例来学习 netcat 命令。
参考链接：https://blog.csdn.net/zheng0518/article/details/41909127
'''
