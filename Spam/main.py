'''
本程序中所使用数据集来自UCI机器学习资源库中的数据集。
地址为https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection。
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB  #离散数据使用多项式朴素贝叶斯
from sklearn.metrics import accuracy_score

#划分训练集、测试集,并生成词袋。
def data_set(data_table):
    label_train, label_test, sms_train, sms_test = train_test_split(data_table['label'],
                                                                    data_table['sms_message'],
                                                                    random_state=1)
    print('Number of rows of the total set:{}'.format(data_table.shape[0]))
    print('Number of rows of the training set:{}'.format(label_train.shape[0]))
    print('Number of rows of the total set:{}'.format(label_test.shape[0]))
    count_vector = CountVectorizer()
    training_data = count_vector.fit_transform(sms_train)
    testing_data = count_vector.transform(sms_test)
    return training_data,testing_data,label_train, label_test

#使用sklearn实现朴素贝叶斯
def calculate(training_data, testing_data, label_train, label_test):
    NB = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True) #平滑处理准确度更高
    NB.fit(training_data, label_train)
    prediction = NB.predict(testing_data)
    print("The accuracy is:{}".format(accuracy_score(label_test, prediction)))

if __name__ == "__main__":
    data_table = pd.read_table("d:/CodeStation/MachineLearning/Spam/smsspamcollection/SMSSpamCollection",
                                sep='\t', names=['label', 'sms_message'])
    #print(data_table.head())    #打印前五行数据
    #print(data_table.shape)  #输出数据表的行、列数
    data_table['label'] = data_table.label.map({'ham': 0, 'spam': 1})  #将标签转化为二元变量，ham为0，spam为1
    #print(data_table.head())
    training_data, testing_data, label_train, label_test = data_set(data_table)
    calculate(training_data, testing_data, label_train, label_test)

