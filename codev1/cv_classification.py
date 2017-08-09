
import os
import numpy as np
import re
from sklearn.decomposition import PCA
from sklearn import svm

f_train_data = open('../data/spatial/conva_path_train01.txt')
f_test_data = open('../data/spatial/conva_path_test01.txt')

# train set
train_data = []
train_label = []
# test set
test_data = []
test_label = []

train_label = np.loadtxt('../data/train_label01.txt')
test_label = np.loadtxt('../data/test_label01.txt')


def read_all_feature(path):
    for index, line in enumerate(path.readlines()):
        file_list = os.listdir(line.strip())
        conv_data = []
        print line.strip()
        file_list = sorted(file_list, key=lambda x: (int(re.sub('\D', '', x)), x))
        for conv in file_list:
            if os.path.splitext(conv)[1] == '.npy':
                conv_path = line.strip() + "/" + conv
                # print conv_path
                data = np.load(conv_path)
                data = np.transpose(np.amax(data, axis=1).reshape((512, -1)))
                # print data.shape
                data = [v for v in data]
                # print len(data),len(data[0])
                conv_data.extend(data)
        print len(conv_data),len(conv_data[0])


if __name__ == '__main__':
    read_all_feature(f_train_data)

