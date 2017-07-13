import numpy as np
import os
from struct import *
from sklearn import svm

# f_train_data = open('../data/spatial/data_path_train01.txt')
# f_test_data = open('../data/spatial/data_path_test01.txt')
#
# # train set
# train_data = []
# train_label = []
# # test set
# test_data = []
# test_label = []
#
# train_label = np.loadtxt('../data/train_label01.txt')
# test_label = np.loadtxt('../data/test_label01.txt')
#
# # all clips feature
# train_split_data = []
# train_split_label = []
#
#
# def calc_accuracy(pre, y):
#     print sum(pre == y), len(y)
#     return float(sum(pre == y))/len(y)
from sklearn.externals import joblib


def read_fc_feature(data_path, label_path):
    train_label = np.loadtxt(label_path)
    f_open = open(data_path)
    train_split_data = []
    train_split_label = []
    for index, line in enumerate(f_open.readlines()):
        file_list = os.listdir(line.strip())
        for fc in file_list:
            if os.path.splitext(fc)[1] == '.fc6-1':
                fc_path = line.strip() + "/" + fc
                # print fc_path
                with open(fc_path, mode='rb') as fid:
                    file_content = fid.read()
                s = unpack("iiiii", file_content[:20])
                m = s[0] * s[1] * s[2] * s[3] * s[4]
                data = []
                for i in range(0, m):
                    start = 20 + (i * 4)
                    d = unpack("f", file_content[start:start + 4])
                    data.append(d[0])
                train_split_data.append(data)
                train_split_label.append(train_label[index])


def read_only_one_featue(data_path, label_path):
    train_label = np.loadtxt(label_path)
    f_open = open(data_path)
    for index, line in enumerate(f_open.readlines()[:100]):
        file_list = os.listdir(line.strip())
        for fc in file_list:
            if os.path.splitext(fc)[1] == '.fc6-1':
                fc_path = line.strip() + "/" + fc
                print fc_path
                with open(fc_path, mode='rb') as fid:
                    file_content = fid.read()
                s = unpack("iiiii", file_content[:20])
                m = s[0] * s[1] * s[2] * s[3] * s[4]
                data = []
                for i in range(0, m):
                    start = 20 + (i * 4)
                    d = unpack("f", file_content[start:start + 4])
                    data.append(d[0])
                print(data)


def read_avg_feature(data_path, label_path):
    train_label = np.loadtxt(label_path)
    train_data = []
    f_open = open(data_path)
    for index, line in enumerate(f_open.readlines()):
        file_list = os.listdir(line.strip())
        for fc in file_list:
            if fc == 'c3d.fc6':
                fc_path = line.strip() + "/" + fc
                print fc_path
                data = []
                with open(fc_path, mode='rb') as fid:
                    file_content = fid.read()
                    for i in range(0, 4096):
                        start = i * 4
                        d = unpack('f', file_content[start:start + 4])
                        data.append(d[0])
                train_data.append(data)
    return train_data, train_label


def train_model(train_x, train_y):
    c = 0.01  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=c).fit(train_x, train_y)
    joblib.dump(svc, 'svm_fc_avg.pkl')
    return svc

if __name__ == '__main__':
    train_x, train_y = read_avg_feature('../data/spatial/data_path_train01.txt', '../data/train_label01.txt')
    test_x, test_y = read_avg_feature('../data/spatial/data_path_test01.txt', '../data/test_label01.txt')
    print("=========data size==========")
    print("train data size {}".format(len(train_x)))
    print("test data size {}".format(len(test_x)))
    print("=========training===========")
    model = train_model(train_x, train_y)
    print("=========testing============")
    model.score(test_x, test_y)
