import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from q3_Partition_and_Evaluation_Metrics import evaluation_metric
rng = np.random.default_rng()

# Load in the data
mnist_ori_data = np.load("data/mnist-data.npz")
field = ("training_data", "training_labels", "test_data")

mnist_data = mnist_ori_data[field[0]]
mnist_label = mnist_ori_data[field[1]]

mnist_tot_num = len(mnist_data)
mnist_idx = rng.permutation(np.arange(mnist_tot_num))
mnist_vali_num = 10000
mnist_train_num = 10000

def partition_data(train_num, vali_num, data, lab, idx):
    vali_idx = idx[:vali_num]
    train_idx = idx[vali_num:vali_num + train_num]
    data_vali = data[vali_idx]
    lab_vali = lab[vali_idx]
    data_train = data[train_idx]
    lab_train = lab[train_idx]
    return data_vali, lab_vali, data_train, lab_train

mnist_data_vali, mnist_label_vali, mnist_data, mnist_label = partition_data(
    mnist_train_num, mnist_vali_num, mnist_data, mnist_label, mnist_idx)

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

mnist_data, mnist_label = mnist_preprocess(mnist_data, mnist_label)
mnist_data_vali, mnist_label_vali = mnist_preprocess(mnist_data_vali, mnist_label_vali)

# hyper-parameter tuning
C_value = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000])

def svm_hyperparameter_tune(data, label, data_vali, label_vali):
    acc_rate = np.zeros(8)
    idx = 0
    for C in C_value:
        clf = LinearSVC(C=C, max_iter=10000)  # 增加迭代次数
        clf.fit(data, label)
        
        true_lab = label_vali
        pred_lab = clf.predict(data_vali)
        vali_acc = evaluation_metric(true_lab, pred_lab)
        acc_rate[idx] = vali_acc
        idx += 1
    return acc_rate

mnist_acc_rate = svm_hyperparameter_tune(mnist_data, mnist_label, mnist_data_vali, mnist_label_vali)
best_idx = np.argmax(mnist_acc_rate)
best_C = C_value[best_idx]

# Plot
def plotting_C(C_val, acc_rate, name, question):
    plt.plot(C_val, acc_rate, label=f"{name} validation accuracy")
    plt.title(f"{name}-linearSVM Accuracy vs Hyperparameter C")
    plt.xlabel("hyper-C")
    plt.xscale("log")
    plt.ylabel("training and validation accuracy")
    plt.legend()
    plt.savefig(f"q{question}_hyper_tuning.png")
plotting_C(C_value, mnist_acc_rate, "MNIST", 5)

