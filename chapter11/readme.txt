the task of this chapter
conception of online learning
stream process with spark stream
online learning with spark
srtuct streaming
online study：
1.stream process：
数据流是连续的记录序列。常见的例子包括网页和移动应用获取的活动流数据，时间戳日志文件，交易数据，甚至传感器或者设备网络传入的事件流。
批处理的方法一般包括数据流报错到一个临时的存储系统如HDFS和在存储数据上运行批处理。为了生成最新的结果，批处理必须在最新的的可用数据上周期性的运行。
相反，流处理方法是当数据产生时就开始处理。
Spark stream介绍：
两种处理技术：
1.单独处理每条记录，并在记录出现时立刻处理
2.把多个记录组合为小批量任务，可以通过记录数或者时间长度划分出来。
spark stream使用第二种方其核心概念是离散化流。一个DStream是指一个小批量作业的序列，每一个小批量作业表示为一个Spark RDD。
离散化流是通过输入数据源和叫做批处理间隔的时间窗口来定义。
数据流被分成和批处理间隔相等的时间段。如果在所给时间段内没有产生数据，将得到一个空的RDD。
1.输入源
Spark Streaming 接收端负责从数据源接收数据并转换成由SPARK RDD 组成Dstream
支持多种输入源：基于文件的源，和基于网络的源（TWITTER api，消息队列，Flume，KAfka，Amazon kimesis）等分布式流及其日志流。
2.转换
spark支持对RDD的转换，Dstream是由RDD组成的，所以也能转换DSTREAM。
SPARK stream中的 reduce和count等算子不是执行算子而是转换算子。
（1）状态转换
可以使用广播变量或者累增变量来并行更新状态
（2）一般转换
提供transform函数，以方便用户访问流中的每个RDD的批量数据。也就是高层api将一个DSTREAM转换成另一个Dstream。不同类型rdd之间进行操作。
3.执行算子
stream中的算子（count）不像批量RDD那样是执行算子，他自己有一套执行算子的概念。比如下面几个：print,saveAsObjectFile,saveAsHadoopFiles
forEachRDD:这个算子用得最多，允许用户对Dstream中的每一个批量数据对应的`RDD本身做任意操作，经常用来产生附加效果，比如数据保存到外部系统，打印测试，导出作为图标。
4.窗口算子
时间窗口，窗口由窗口长度和滑动间隔定义。例如：10秒的窗口，5秒的间隔可以定义一个窗口，它每5秒计算一次前10s接收的Dstream数据。
Spark缓存机制和容错机制
和Spark的RDD一样，DStream也可以缓存在内存里。缓存的使用场景也和RDD类似。如果需要多次访问DStream中的数据，缓存会带来很大好处。
RDD是不可变的数据集合，并由输入数据源和类群定义。类群，就是应用到RDD上的转换算子和执行算子的集合。RDD中的容错就是重建因工作节点故障而丢失的RDD。
