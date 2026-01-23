import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from q3_Partition_and_Evaluation_Metrics import evaluation_metric
rng = np.random.default_rng()

# Load in the data
mnist_ori_data = np.load("data/mnist-data.npz")
spam_ori_data = np.load("data/spam-data.npz")

field = ("training_data", "training_labels", "test_data")

mnist_data = mnist_ori_data[field[0]]
mnist_label = mnist_ori_data[field[1]]
spam_data = spam_ori_data[field[0]]
spam_label = spam_ori_data[field[1]]

mnist_tot_num = len(mnist_data)
mnist_idx = rng.permutation(np.arange(mnist_tot_num))
mnist_train_num = [100, 200, 500, 1000, 2000, 5000, 10000]
mnist_vali_num = 10000

spam_tot_num = len(spam_data)
spam_idx = rng.permutation(np.arange(spam_tot_num))
spam_vali_num = int(spam_tot_num * 0.2)
spam_train_max = spam_tot_num - spam_vali_num  # 训练集最大样本数
spam_train_num = [100, 200, 500, 1000, 2000, spam_train_max]

def partition_data(vali_num, data, lab, idx):
    """用随机索引分割验证集和训练集，避免重叠"""
    vali_idx = idx[:vali_num]
    train_idx = idx[vali_num:]
    data_vali = data[vali_idx]
    lab_vali = lab[vali_idx]
    data_train = data[train_idx]
    lab_train = lab[train_idx]
    return data_vali, lab_vali, data_train, lab_train

mnist_data_vali, mnist_label_vali, mnist_data, mnist_label = partition_data(
    mnist_vali_num, mnist_data, mnist_label, mnist_idx)
spam_data_vali, spam_label_vali, spam_data, spam_label = partition_data(
    spam_vali_num, spam_data, spam_label, spam_idx)

# Preprocess the data
def mnist_preprocess(dat, lab):
    """
    normalize the mnist data set
    dat(array): original data
    lab(array): labels
    """
    dat = dat.reshape(dat.shape[0], -1)
    dat = dat.astype(np.float32) / 255.0
    lab = lab.astype(np.int64).ravel()
    return dat, lab

def spam_preprocess(dat, lab):
    """
    preprocess spam data
    dat(array): original data
    lab(array): labels
    """
    dat = dat.reshape(dat.shape[0], -1)
    dat = dat.astype(np.float32)
    lab = lab.astype(np.int64).ravel()
    return dat, lab

mnist_data, mnist_label = mnist_preprocess(mnist_data, mnist_label)
mnist_data_vali, mnist_label_vali = mnist_preprocess(mnist_data_vali, mnist_label_vali)

spam_data, spam_label = spam_preprocess(spam_data, spam_label)
spam_data_vali, spam_label_vali = spam_preprocess(spam_data_vali, spam_label_vali)

# SVM
def train_svm(dat, lab):
    clf = LinearSVC(C=0.01)
    clf.fit(dat, lab)
    return clf

def train_and_eval(train_dat, train_lab, vali_dat, vali_lab):
    clf = train_svm(train_dat, train_lab)
    true_train_lab = train_lab
    pred_train_lab = clf.predict(train_dat)
    true_vali_lab = vali_lab
    pred_vali_lab = clf.predict(vali_dat)
    train_acc = evaluation_metric(true_train_lab, pred_train_lab)
    vali_acc = evaluation_metric(true_vali_lab, pred_vali_lab)
    return train_acc, vali_acc

# Main
def plotting(train_num, data, label, data_vali, label_vali, name):
    """
    draw the plot
    name: "MNIST" or "spam"
    """
    train_acc = np.zeros(len(train_num))
    vali_acc = np.zeros(len(train_num))
    dep = 0
    for i in train_num:
        cur_dat = data[:i]
        cur_lab = label[:i]
        train_acc[dep], vali_acc[dep] = train_and_eval(cur_dat, cur_lab, data_vali, label_vali)
        dep += 1
    plt.clf()
    plt.plot(train_num, train_acc, label=f"{name} training accuracy")
    plt.plot(train_num, vali_acc, label=f"{name} validation accuracy")
    plt.title(f"{name}-linearSVM Accuracy vs Training Size")
    plt.xlabel("number of training examples")
    plt.ylabel("training and validation accuracy")
    plt.legend()
    plt.savefig(f"q4_best_{name}.png")

plotting(mnist_train_num, mnist_data, mnist_label, mnist_data_vali, mnist_label_vali, "MNIST")
plotting(spam_train_num, spam_data, spam_label, spam_data_vali, spam_label_vali, "spam")
