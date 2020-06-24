"""
自己实现的朴素贝叶斯算法
"""
import numpy as np
from collections import Counter
import numpy as np
import csv
from math import exp
from math import sqrt
from math import pi

def load_data(file_name,gen):
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
        i = 0
        if gen == 0:
            while i < sem:
                test_data.append(lib_data[test_idx[i]])
                test_label.append(lib_label[test_idx[i]])
                i += 1
        elif gen == 1:
            while i < sem:
                if lib_label[test_idx[i]] == 1:
                    test_data.append(lib_data[test_idx[i]])
                    test_label.append(lib_label[test_idx[i]])
                i += 1
        elif gen == 2 :
            while i < sem:
                if lib_label[test_idx[i]] == 0:
                    test_data.append(lib_data[test_idx[i]])
                    test_label.append(lib_label[test_idx[i]])
                i += 1
    return train_data, test_data, train_label, test_label


class GaussianNB:
    def __init__(self):
        self.prior = None
        self.avgs = None
        self.vars = None
        self.n_class = None

    '''
    计算先验概率
    '''
    def _get_prior(self, label):
        cnt = Counter(label)
        prior = np.array([cnt[i] / len(label) for i in range(len(cnt))])
        return prior

    '''
    每个label分别计算训练集均值
    '''
    def _get_avgs(self, data, label):
        male_data, female_data, avgs_data = [], [], []
        np.array(male_data)
        np.array(female_data)
        np.array(avgs_data)
        for i in range(len(label)):
            if label[i] == 1:
                male_data.append(data[i])
            else: female_data.append(data[i])
        avgs_data.append(np.mean(male_data,0))
        avgs_data.append(np.mean(female_data, 0))
        return avgs_data

    '''
    每个label分别计算训练集方差
    '''
    def _get_vars(self, data, label):
        male_data, female_data, vars_data= [], [], []
        for i in range(len(label)):
            if label[i] == 1:
                male_data.append(data[i])
            else: female_data.append(data[i])
        vars_data.append(np.var(male_data,0))
        vars_data.append(np.var(female_data, 0))
        return vars_data

    '''
    计算似然度
    '''
    def _get_likelihood(self, row):
        result= []
        for i in range(self.n_class):
            atom = []
            for j in range(len(row)):
                temp = (1 / sqrt(2 * pi * self.vars[i][j]) * exp(-(row[j] -
                self.avgs[i][j]) ** 2 / (2 * self.vars[i][j])))
                atom.append(temp)
            result.append(atom)
        result = np.prod(result, 1)
        return result

    def _get_acc(self, label_test, label_pre):
        count = 0;
        for i in range(len(label_test)):
            if label_test[i] != label_pre[i]:
                count += 1
        return count / len(label_test)    

    '''
    训练模型
    '''
    def fit(self, data, label):
        self.prior = self._get_prior(label)
        self.n_class = len(self.prior)
        self.avgs = self._get_avgs(data, label)
        self.vars = self._get_vars(data, label)

    '''
    用先验概率乘以似然度再归一化得到每个label的prob。
    '''
    def predict_prob(self, data):
        likelihood = np.apply_along_axis(self._get_likelihood, axis=1, arr=data)
        probs = self.prior * likelihood
        probs_sum = probs.sum(axis=1)
        result = []
        for i in range(len(probs)):
            result.append(probs[i] / probs_sum[i])
        return np.array(result)

    def predict(self, data):
        return self.predict_prob(data).argmax(axis=1)

if __name__ == "__main__":
    file_name = "voice.csv"
    print("When gender unexprcted:")
    print("--------------------------------------------")
    data_train, data_test, label_train, label_test = load_data(file_name, 0)
    clf = GaussianNB()
    clf.fit(data_train, label_train)
    label_pre = clf.predict(data_test)
    acc = clf._get_acc(label_test, label_pre)
    print("Accuracy is %.3f" % acc)
    print("---------------------------------------------")

    print("when focus on male:")
    print("--------------------------------------------")
    data_train, data_test, label_train, label_test = load_data(file_name, 1)
    clf = GaussianNB()
    clf.fit(data_train, label_train)
    label_pre = clf.predict(data_test)
    acc = clf._get_acc(label_test, label_pre)
    print("Accuracy is %.3f" % acc)
    print("---------------------------------------------")

    print("when focus on female:")
    print("--------------------------------------------")
    data_train, data_test, label_train, label_test = load_data(file_name, 2)
    clf = GaussianNB()
    clf.fit(data_train, label_train)
    label_pre = clf.predict(data_test)
    acc = clf._get_acc(label_test, label_pre)
    print("Accuracy is %.3f" % acc)
    print("---------------------------------------------")

