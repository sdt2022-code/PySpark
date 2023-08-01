# Databricks notebook source
import urllib.request
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
import matplotlib.pyplot as plt

from pyspark.sql.functions import mean , stddev
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.classification import SVMWithSGD
from pyspark.ml import Pipeline
from pyspark.mllib.linalg import Vector as MLLibVector, Vectors as MLLibVectors
from pyspark.mllib.evaluation import BinaryClassificationMetrics 
from pyspark.mllib.regression import LabeledPoint


# COMMAND ----------

#urllib.request.urlretrieve("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz", "/tmp/kddcup_data.gz")
#dbutils.fs.mv("file:/tmp/kddcup_data.gz", "dbfs:/kdd/kddcup_data.gz")
#display(dbutils.fs.ls("dbfs:/kdd"))

# COMMAND ----------

#Creating a Spark Session
spark = SparkSession.builder.appName("Assignment3").getOrCreate()

# COMMAND ----------

#Importing the data from local directory. 

file_path = 'dbfs:/FileStore/shared_uploads/saridtarabay@gmail.com/kddcup_data_10_percent.gz'
data_rdd = sc.textFile(file_path)

# COMMAND ----------

#display(data_rdd)
data_rdd.collect()

# COMMAND ----------

type(data_rdd)

# COMMAND ----------

#distinct types in RDD

dis_types = data_rdd.map(lambda x: type(x)).distinct().collect()

for entry_type in dis_types:
    print(entry_type)

#This is giving type string because each datapoint is represented by a string seperated by commas for each feature. 


# COMMAND ----------

#PART 4: Splitting the data
data_rdd_split = data_rdd.map(lambda line: line.split(','))

num_of_features = len(data_rdd_split.take(1)[0])
# can also do num_of_features  = data_rdd_split.map(lambda row: len(row)).distinct().collect() in case of unstructed dataset
print('Total number of features is equal to:' , num_of_features)


# COMMAND ----------

# Checking for type of each data entry:

columns = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins',
 'logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds', 'is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label']

data_df = spark.createDataFrame(data_rdd_split , columns)
display(data_df)

# COMMAND ----------

# Fixing Datatypes:

str_columns = ['protocol_type','service','flag','land','logged_in','is_host_login','is_guest_login']

for c in data_df.columns:
    if c in str_columns:
        data_df = data_df.withColumn(c , col(c).cast("string"))
    elif c == 'label':
        data_df = data_df.withColumn(c , col(c))
    else:
        data_df = data_df.withColumn(c , col(c).cast("long"))



# COMMAND ----------

data_df.printSchema()

# COMMAND ----------

# PART 5

columns_of_interest = ['duration', 'protocol_type', 'service' , 'src_bytes', 'dst_bytes', 'flag', 'label']
data_of_interest = data_df.select(columns_of_interest)
data_of_interest_rdd = data_of_interest.rdd
data_of_interest.printSchema()
display(data_of_interest.take(10))

# COMMAND ----------

data_of_interest_rdd.take(10)

# COMMAND ----------

# PART 6
protocol_type_connection_num = (data_of_interest.groupBy('protocol_type').count()).sort('count')
protocol_type_connection_num.show()

# COMMAND ----------

service_type_connection_num = (data_of_interest.groupBy('service').count()).sort('count')
service_type_connection_num.show()

# COMMAND ----------

# plotting the bar plots
service_data_pd_df = service_type_connection_num.toPandas()
plt.figure(figsize = (20,5))
plt.bar(service_data_pd_df['service'] , service_data_pd_df['count'])
plt.xlabel('Service')
plt.xticks(rotation = 90)
plt.ylabel('Count')
plt.title('Connections per service')

# COMMAND ----------

protocol_type_pd_df = protocol_type_connection_num.toPandas()
plt.figure(figsize = (10,5))
plt.bar(protocol_type_pd_df['protocol_type'] , protocol_type_pd_df['count'])
plt.xlabel('Protocol')
plt.xticks(rotation = 90)
plt.ylabel('Count')
plt.title('Connections per protocol')

# COMMAND ----------

# PART 7

# Visualising services that get attacked the most. 

service_normal_num = data_of_interest.filter(data_of_interest.label == 'normal.').select('service').groupBy('service').count().toPandas()
service_attack_num = data_of_interest.filter(data_of_interest.label != 'normal.').select('service').groupBy('service').count().toPandas()

# COMMAND ----------

plt.figure(figsize = (20,7))
plt.bar(service_normal_num['service'] , service_normal_num['count'] , color = 'blue',label = 'Normal')
plt.bar(service_attack_num['service'] , service_attack_num['count'] , color = 'red', label = 'Attack')
plt.xlabel('Service')
plt.legend()
plt.xticks(rotation = 90)
plt.title('Attacks per service')
plt.show()

# COMMAND ----------

# Visualizing the distribution of srcand dst bytes for regular and attack connections.


# Normal connections
normal_connections = data_of_interest.filter(data_of_interest['label']=='normal.')
src_bytes_normal = normal_connections.select('src_bytes')
dst_bytes_normal = normal_connections.select('dst_bytes')

src_bytes_normal_mean = src_bytes_normal.select(mean('src_bytes')).collect()[0][0]
src_bytes_normal_std = src_bytes_normal.select(stddev('src_bytes')).first()[0]

src_bytes_step1 = src_bytes_normal.withColumn('subtracted_column', col('src_bytes') - src_bytes_normal_mean)

# UDF to divide by std
divide_src_normal_udf = udf(lambda x: x / src_bytes_normal_std, FloatType())

src_bytes_normal_stand = src_bytes_step1.withColumn('stand_src_column', divide_src_normal_udf(src_bytes_step1['subtracted_column']))


dst_bytes_normal_avg = dst_bytes_normal.select(mean('dst_bytes')).collect()[0][0]
dst_bytes_normal_std = dst_bytes_normal.select(stddev('dst_bytes')).first()[0]


dst_bytes_step1 = dst_bytes_normal.withColumn('subtracted_column', col('dst_bytes') - dst_bytes_normal_avg)
divide_dst_normal_udf = udf(lambda x: x / dst_bytes_normal_std, FloatType())
dst_bytes_normal_stand = dst_bytes_step1.withColumn('stand_dst_column', divide_dst_normal_udf(dst_bytes_step1['subtracted_column']))



# Attack connections
attack_connections = data_of_interest.filter(data_of_interest['label'] != 'normal.')
src_bytes_attack = attack_connections.select('src_bytes')
dst_bytes_attack = attack_connections.select('dst_bytes')

src_bytes_attack_avg = src_bytes_attack.select(mean('src_bytes')).collect()[0][0]
src_bytes_attack_std = src_bytes_attack.select(stddev('src_bytes')).first()[0]


src_bytes_a_step1 = src_bytes_attack.withColumn('subtracted_column', col('src_bytes') - src_bytes_attack_avg)

# UDF to divide by std
divide_src_attack_udf = udf(lambda x: x / src_bytes_attack_std, FloatType())

src_bytes_attack_stand = src_bytes_a_step1.withColumn('stand_src_column', divide_src_attack_udf(src_bytes_a_step1['subtracted_column']))
                                                    


dst_bytes_attack_mean = dst_bytes_attack.select(mean('dst_bytes')).collect()[0][0]
dst_bytes_attack_std = dst_bytes_attack.select(stddev('dst_bytes')).first()[0]


dst_bytes_a_step1 = dst_bytes_attack.withColumn('subtracted_column', col('dst_bytes') - dst_bytes_attack_mean)
divide_dst_attack_udf = udf(lambda x: x / dst_bytes_attack_std, FloatType())
dst_bytes_attack_stand = dst_bytes_a_step1.withColumn('stand_dst_column', divide_dst_attack_udf(dst_bytes_a_step1['subtracted_column']))



# COMMAND ----------

src_normal_to_series = src_bytes_normal_stand.toPandas()['stand_src_column']
src_attack_to_series = src_bytes_attack_stand.toPandas()['stand_src_column']

plt.figure(figsize = (10,5))
plt.hist(src_normal_to_series ,  bins = 10000 , alpha = 0.5, label = 'src_bytes normal' , color = 'blue')
plt.hist(src_attack_to_series , bins = 20000 , alpha = 0.5, label = 'src_bytes attack' , color = 'red')
plt.title('src_bytes distribution')
plt.xlim(-0.5, 1)
plt.legend()
plt.show()

# COMMAND ----------

dst_normal_to_series = dst_bytes_normal_stand.toPandas()['stand_dst_column']
dst_attack_to_series = dst_bytes_attack_stand.toPandas()['stand_dst_column']

plt.figure(figsize = (10,5))
plt.hist(dst_normal_to_series ,  bins = 10000 , alpha = 0.5, label = 'dst_bytes normal' , color = 'blue')
plt.hist(dst_attack_to_series , bins = 20000 , alpha = 0.5, label = 'dst_bytes attack' , color = 'red')
plt.title('dst_bytes distribution')
plt.xlim(-0.5, 1)
plt.legend()
plt.show()

# Values of dst are scaled to approximately between [0,1] due to precision, but I think the offset is being exagerrated on the graph

# COMMAND ----------

# compute the covariance matrix for a set of features including the label of attack or normal conenction

string_feature_column1 = 'protocol_type'
string_feature_column2 = 'service'
string_label = 'label'

# Create a StringIndexer to encode the string feature column
indexer1 = StringIndexer(inputCol=string_feature_column1, outputCol='indexed_protocol')
indexed_df1 = indexer1.fit(data_of_interest).transform(data_of_interest)

indexer2 = StringIndexer(inputCol=string_feature_column2, outputCol='indexed_service')
indexed_df2 = indexer2.fit(indexed_df1).transform(indexed_df1)

indexer3 = StringIndexer(inputCol=string_label, outputCol='indexed_label')
indexed_df3 = indexer3.fit(indexed_df2).transform(indexed_df2)

# Show the encoded DataFrame
indexed_df3.show()


# COMMAND ----------


# looking at correlation of features selected in part 5 
# This was done to see the relative importance of features on the connectino type. 
# Note: usually it is better to standardize, however standardizing the indexed columns would not be representative. 

columns_list = ['duration','src_bytes' , 'dst_bytes','indexed_protocol' , 'indexed_service','indexed_label']
data_for_corr = indexed_df3.select(columns_list).toPandas()

# COMMAND ----------

corr_mat = data_for_corr.corr()

# COMMAND ----------

corr_mat

# COMMAND ----------

# Looking at normal vs attack connections when logged in

attack_login = data_df.filter(data_df['label'] == 'normal.').select('logged_in')
count_normal = attack_login.count()

print('The number of regular connections when logged in is:' , count_normal)

attack_login_2 = data_df.filter(data_df['label'] != 'normal.').select('logged_in')
count_attack = attack_login_2.count()

print('The number of attack connections when logged in is:' , count_attack)


# COMMAND ----------

plt.figure()
plt.bar(['normal'] , attack_login.count(),label = 'test1')
plt.bar(['attack'] , attack_login_2.count() , label = 'attakc')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **MACHINE LEARNING MODEL**

# COMMAND ----------

df_modified = data_df.withColumn('label',when(data_df.label == 'normal.',lit('normal')).otherwise(lit('attack')))


# COMMAND ----------

df_modified.printSchema()

# COMMAND ----------

features_for_ml = ['duration','protocol_type','service','src_bytes','dst_bytes','logged_in', 'count','srv_count','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','label']
df_for_ml = df_modified.select(features_for_ml)


# COMMAND ----------

display(df_for_ml)

# COMMAND ----------

cat_cols = ['protocol_type','service','logged_in']
num_cols = [x for x in df_for_ml.columns if x not in cat_cols]
num_cols.remove('label')

# COMMAND ----------

num_cols

# COMMAND ----------

#indexing string variables
ind_stages = []

for col in cat_cols:
    indexer = StringIndexer(inputCol = col , outputCol = col +' Indexed')
    encoder = OneHotEncoder(inputCols=[indexer.getOutputCol()],outputCols=[col + "classVec"])
    ind_stages += [indexer, encoder]


label_idx =  StringIndexer(inputCol="label", outputCol="newlabel")
ind_stages += [label_idx]
Inputs = [c + "classVec" for c in cat_cols] + num_cols
assembler = VectorAssembler(inputCols = Inputs , outputCol = 'features')

ind_stages += [assembler]

# COMMAND ----------

pipeline = Pipeline().setStages(ind_stages)
pipelineModel = pipeline.fit(df_for_ml)
model = pipelineModel.transform(df_for_ml)

# COMMAND ----------

display(model)

# COMMAND ----------

df_for_ml_transformed = model.select('newlabel' , 'features')
df_for_ml_transformed = df_for_ml_transformed.selectExpr("features as features", "newlabel as label")


# COMMAND ----------

display(df_for_ml_transformed)


# COMMAND ----------

# Splitting the data into training and testing
train_data , test_data = df_for_ml_transformed.randomSplit([0.8,0.2] , seed = 12)

# COMMAND ----------

#Standardizing the data
scaler = StandardScaler(inputCol = 'features' , outputCol = 'stand_features')
scaler_model = scaler.fit(train_data.select('features'))
train_data = scaler_model.transform(train_data)
test_data = scaler_model.transform(test_data)

# COMMAND ----------

display(train_data)


# COMMAND ----------

display(test_data)

# COMMAND ----------

# First Machine Learning Algorithm (SVM)

SVM_model = SVMWithSGD.train(train_data.rdd.map(lambda row: LabeledPoint(row.label, MLLibVectors.fromML(row.stand_features))), iterations=10, regParam=0.5)
predictions = test_data.rdd.map(lambda d: ((d.label), float(SVM_model.predict(MLLibVectors.fromML(d.stand_features)))))

predictions.take(10)


# COMMAND ----------

# computing the accuracy
wrong_pred_num = predictions.filter(lambda x: int(x[0] != int(x[1]))).count()
total_pred_num = predictions.count()

print('Error rate : ', float(wrong_pred_num) / float(total_pred_num))
print('Total accuracy : ', 1 - float(wrong_pred_num) / float(total_pred_num))

#Area under ROC
metric = BinaryClassificationMetrics(predictions)
print('The total AUC is : ' , metric.areaUnderROC)

#Area under PR curve
print('Area under PR curve is: ', metric.areaUnderPR)

# COMMAND ----------

# Implementating a Logisitc Regression model 

lr = LogisticRegression(labelCol = 'label' , featuresCol = 'stand_features')
model = lr.fit(train_data.select('label' , 'stand_features'))
pred = model.transform(test_data)



# COMMAND ----------

pred.show()

# COMMAND ----------

#Evaluating model performance

evaluator = BinaryClassificationEvaluator(labelCol = 'label')
evaluator_pr = BinaryClassificationEvaluator(labelCol='label', metricName='areaUnderPR')

# Accuracy of the model

wrong_pred = pred.filter(pred['prediction'] != pred['label']).count()
total_pred_num_log = pred.count()

error_rate = float(wrong_pred) / float(total_pred_num_log)
print('The error rate of the logisitic regression model is: ', error_rate)
print('The accruacy of the logistic regression model is: ', 1-error_rate )


#Evaluating AUC score (under ROC curve)
auc = evaluator.evaluate(pred)
print('The AUC score of the logistic regression model is:' , auc)

#Area under PR curve
au_pr = evaluator_pr.evaluate(pred)
print('The Area under the PR curve of the logistic regression model is :' , au_pr)



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Answer to Question 9**
# MAGIC
# MAGIC * Due to the nature of the problem, I chose to compare the performance of a simple algorithm, i.e., the Logstic Regression algorithm (binary classification) to a slightly more complex algorithm, i.e., the SVM (Support Vector Machine) Algorithm. 
# MAGIC
# MAGIC * The metrics chosen to compare the performance of both algorithms are: error rate, accuracy , AUC score and area under PR curve. 
# MAGIC
# MAGIC * The best performance was the logistic regression model performance achieving an accuracy score of 0.99 compared to 0.97 for SVM, an AUC score of 0.999 (compared to 0.968 for SVM) and and area under the PR score of 0.996 (compared to 0.877 for SVM).

# COMMAND ----------

# MAGIC %md
# MAGIC **PART B**
# MAGIC
# MAGIC 1.  
# MAGIC * A platform as a service (PaaS) solution that hosts web apps in Azure provide professional development services to continuously add features to custom applications : **YES** , PaaS allows easy development and deployement of applications on the cloud based by providing some pre-developed services, resources and tools.
# MAGIC *  A platform as a service (PaaS) database offering in Azure provides built-in high availability: **YES** , PaaS provides high global availability, scalability and fully managed infrastructure to maintian a continuous serive and operation level. 
# MAGIC
# MAGIC 2. **D - Strong consistency guarantees are required** Relational databases or are generally used with a predifined format of the data as well as a predifined relation between tables, generally referenced by a key. This is not the case for non-relational databases which are generally used without a predefined schema, and to store data in real time. 
# MAGIC
# MAGIC 3. **D -  Configure the SaaS solution** When dealing with a SaaS solution, you are only required to configure the software solution. Infrastructure, availability and everything else is managed by the cloud provider. 
# MAGIC
# MAGIC 4. 
# MAGIC * To achieve a hybrid cloud model, a company must always migrate from a private cloud mode: **NO** A company can chose what data to migrate to public cloud and what data to keep on private cloud, or on premise. 
# MAGIC
# MAGIC * A company can extend the capacity of its internal network by using a public cloud: **YES** A company can use the public cloud to extend its compute or storage capacity.
# MAGIC
# MAGIC * In a public cloud model, only guest users at your company can access the resources in the cloud: **NO** Anyone provided permission to access the ressource group on the cloud can access it. 
# MAGIC
# MAGIC 5. 
# MAGIC
# MAGIC a. A cloud service that remains available after a failure occurs **Fault tolerance**, this is the case when a server is down or under maintenance, the user can duplicate his data on use the same resources from another server. 
# MAGIC
# MAGIC b. A cloud service that can be recovered after a failure occurs **Disaster Recovery**, this is the ability of a cloud provider to recover its users data after a failure occurs.
# MAGIC
# MAGIC c. A cloud service that performs quickly when demand increases **Dynamic Scalability**, this is a characteristic of a cloud service being scalable and elastic with increasing or decreasing demand
# MAGIC
# MAGIC d. A cloud service that can be accessed quickly from the internet **Low latency**, this is the distribution and availability of a cloud service in multiple regions around the globe to provide low latency connection to its users. A lower latency means a faster connection.
# MAGIC
# MAGIC
# MAGIC
