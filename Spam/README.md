# 基于朴素贝叶斯的垃圾邮件分类
本程序中所使用数据集来自UCI机器学习资源库中的[数据集](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)  
数据位于[*smsspamcollection*](https://github.com/amber-yangcn/MachineLearning/tree/master/Spam/smsspamcollection)文件夹中，数据文件名为[*SMSSpamCollection*](https://github.com/amber-yangcn/MachineLearning/blob/master/Spam/smsspamcollection/SMSSpamCollection)。  
其内容如下：  
>- ham   What you doing?how are you?  
>- ham   Ok lar... Joking wif u oni...  
>- ham   dun say so early hor... U c already then say...  
>- ham   MY NO. IN LUTON 0125698789 RING ME IF UR AROUND! H*  
>- ham   Siva is in hostel aha:-.  
>- ham   Cos i was out shopping wif darren jus now n i called him 2 ask wat present he wan lor. Then he started guessing who i was wif n he finally guessed darren lor.  
>- spam   FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop  
>- spam   Sunshine Quiz! Win a super Sony DVD recorder if you canname the capital of Australia? Text MQUIZ to 82277. B  
>- spam   URGENT! Your Mobile No 07808726822 was awarded a L2,000 Bonus Caller Prize on 02/09/03! This is our 2nd attempt to contact YOU! Call 0871-872-9758 BOX95QU  
---  
文件[*BagofWord.py*](https://github.com/amber-yangcn/MachineLearning/blob/master/Spam/Bag%20of%20Words.py)为自己实现的词袋模型，可用于与sklearn库中的模型进行比较。  
文件[*main.py*](https://github.com/amber-yangcn/MachineLearning/blob/master/Spam/main.py)即为使用sklearn库实现朴素贝叶斯的主体程序。
