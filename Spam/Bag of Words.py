'''
自己实现的Bag of Words功能，也即sklearns count vectorizer方法
'''
import pprint
import string
from collections import Counter
#文档集合
documents = ['Hello, how are you!',
            'Win money, win from home.',
            'Call me now.',
            'Hello, Call hello you tomorrow?']

#将各文档内容小写化
lower_documents = []
for i in documents:
    i = i.lower()
    lower_documents.append(i)
print(lower_documents)

#删除标点符号
sans_documents = []
for i in lower_documents:
    s = str.maketrans('','', string.punctuation)
    i = i.translate(s)
    sans_documents.append(i)
print(sans_documents)

#令牌化
preprocessed_documents = []
for i in sans_documents:
    i = i.split(' ')
    preprocessed_documents.append(i)
print(preprocessed_documents)

#计算频率
frequency_list = []
for i in preprocessed_documents:
    frequency_list.append(Counter(i))
print(frequency_list)