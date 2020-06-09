import numpy as np
import csv
from sklearn.naive_bayes import GaussianNB  #使用高斯朴素贝叶斯
from sklearn.metrics import accuracy_score
"""
基于贝叶斯公式完成男女声音的识别
"""

def load_data(file_name, gen):
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


#使用sklearn实现朴素贝叶斯
def calculate(train_data, test_data, label_train, label_test):
    NB = GaussianNB()
    NB.fit(train_data, label_train)
    prediction = NB.predict(test_data)
    result = accuracy_score(label_test, prediction)
    return result    


if __name__ == "__main__":

    file_name = "voice.csv"
    print("When gender unexprcted:")
    print("--------------------------------------------")
    train_data, test_data, train_label, test_label = load_data(file_name, 0)
    result1 = calculate(train_data, test_data, train_label, test_label)
    print("The 1st result:[", result1, "]")
    train_data, test_data, train_label, test_label = load_data(file_name, 0)
    result2 = calculate(train_data, test_data, train_label, test_label)
    print("The 2nd result:[", result2, "]")
    train_data, test_data, train_label, test_label = load_data(file_name, 0)
    result3 = calculate(train_data, test_data, train_label, test_label)
    print("The 3rd result:[",result3,"]")
    result = (result1+result2+result3)/3
    print("The average:[",result,"]")
    print("---------------------------------------------")

    print("when focus on male:")
    print("--------------------------------------------")
    train_data, test_data, train_label, test_label = load_data(file_name, 1)
    result1 = calculate(train_data, test_data, train_label, test_label)
    print("The 1st result:[", result1, "]")
    train_data, test_data, train_label, test_label = load_data(file_name, 1)
    result2 = calculate(train_data, test_data, train_label, test_label)
    print("The 2nd result:[", result2, "]")
    train_data, test_data, train_label, test_label = load_data(file_name, 1)
    result3 = calculate(train_data, test_data, train_label, test_label)
    print("The 3rd result:[",result3,"]")
    result = (result1+result2+result3)/3
    print("The average:[",result,"]")
    print("---------------------------------------------")

    print("When focus on female:")
    print("--------------------------------------------")
    train_data, test_data, train_label, test_label = load_data(file_name, 2)
    result1 = calculate(train_data, test_data, train_label, test_label)
    print("The 1st result:[", result1, "]")
    train_data, test_data, train_label, test_label = load_data(file_name, 2)
    result2 = calculate(train_data, test_data, train_label, test_label)
    print("The 2nd result:[", result2, "]")
    train_data, test_data, train_label, test_label = load_data(file_name, 2)
    result3 = calculate(train_data, test_data, train_label, test_label)
    print("The 3rd result:[",result3,"]")
    result = (result1+result2+result3)/3
    print("The average:[",result,"]")
    print("---------------------------------------------")

