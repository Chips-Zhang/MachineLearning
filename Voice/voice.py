import numpy as np
import csv
from sklearn.naive_bayes import GaussianNB  #离散数据使用多项式朴素贝叶斯
from sklearn.metrics import accuracy_score
"""
基于贝叶斯公式完成男女声音的识别
"""

def load_data(file_name):
    """
    :param filename
    :return

    """
    data = []
    lib_label = []
    with open(file_name) as file:
        voice = csv.DictReader(file)
        title_name = list(voice.fieldnames)
        title_num = len(title_name) - 1
        for line in voice.reader:
            data.append(line[:title_num])
            # 将性别二值化，male：1，female：0，并存入label
            if line[-1] == 'male':
                gender = 1
            else:
                gender = 0
            lib_label.append(gender)

        # 计算平均值，代替列表中的缺失值
        data = np.array(data).astype(float)
        sum_vec = np.sum(data, axis=0)
        num_vec = np.count_nonzero(data, axis=0)
        mean_vec = sum_vec / num_vec

        for row in range(len(data)):
            for col in range(title_num):
                if data[row][col] == 0.0:
                    data[row][col] = mean_vec[col]

        #将各项数据离散化
        '''
        min_vec = data.min(axis=0)
        max_vec = data.max(axis=0)
        diff_vec = max_vec - min_vec
        diff_vec /= 32  #离散范围
        lib_data = []
        for row in range(len(data)):
            line = np.array((data[row] - min_vec) / diff_vec).astype(int)
            line = list(line)
            lib_data.append(line)
        '''

        lib_data = data
        # 划分数据集
        train_data, test_data, train_label, test_label= [], [], [], []
        lib_data_num = len(lib_data)
        idx = np.random.permutation(lib_data_num)
        div_line = int(0.7 * lib_data_num)
        train_idx, test_idx = idx[:div_line], idx[div_line:]
        for i in range(div_line): 
            train_data.append(lib_data[train_idx[i]])
            train_label.append(lib_label[train_idx[i]])
        sem = lib_data_num - div_line
        for i in range(sem):
            test_data.append(lib_data[train_idx[i]])
            test_label.append(lib_label[train_idx[i]])
    return train_data, test_data, train_label, test_label


#使用sklearn实现朴素贝叶斯
def calculate(train_data, test_data, label_train, label_test):
    NB = GaussianNB()
    NB.fit(train_data, label_train)
    prediction = NB.predict(test_data)
    print("The accuracy is:{}".format(accuracy_score(label_test, prediction)))      


if __name__ == "__main__":
    
    file_name = "voice.csv"
    train_data, test_data, train_label, test_label = load_data(file_name)
    calculate(train_data, test_data, train_label, test_label)

