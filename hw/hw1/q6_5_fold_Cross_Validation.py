import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from q3_Partition_and_Evaluation_Metrics import evaluation_metric
rng = np.random.default_rng()

# Load in the data
spam_ori_data = np.load("data/spam-data.npz")
field = ("training_data", "training_labels", "test_data")

spam_data = spam_ori_data[field[0]]
spam_label = spam_ori_data[field[1]]

spam_tot_num = len(spam_data)
spam_idx = rng.permutation(np.arange(spam_tot_num))
k = 5
spam_vali_num = int(spam_tot_num) // k

# Preprocess the data
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
spam_data, spam_label = spam_preprocess(spam_data, spam_label)

# Partition
spam_folds_data = [spam_data[spam_idx[i*spam_vali_num:(i+1)*spam_vali_num]] for i in range(k)]
spam_folds_label = [spam_label[spam_idx[i*spam_vali_num:(i+1)*spam_vali_num]] for i in range(k)]

def current_partition(indice, dat):
    """
    dat: spam_folds_data or spam_folds_label
    """
    cat = np.concatenate(dat[:indice] + dat[indice + 1:], axis=0)
    return dat[indice], cat

# train_svm & mean_acc
C_value = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
C_mean_acc = np.zeros(8)
idx = 0
for c in C_value:
    cur_acc = np.zeros(5)
    clf = LinearSVC(C=c, max_iter=10000)
    for i in range(5):
        cur_vali_dat, cur_train_dat = current_partition(i, spam_folds_data)
        cur_vali_lab, cur_train_lab = current_partition(i, spam_folds_label)
        clf.fit(cur_train_dat, cur_train_lab)
        true_lab = cur_vali_lab
        pred_lab = clf.predict(cur_vali_dat)
        cur_acc[i] = evaluation_metric(true_lab, pred_lab)
    C_mean_acc[idx] = np.mean(cur_acc)
    idx += 1

# plotting
plt.plot(C_value, C_mean_acc, label="5-fold CV")
plt.title(f"spam-linearSVM Accuracy vs Hyperparameter C")
plt.xlabel("hyper-C")
plt.xscale("log")
plt.ylabel("training and validation accuracy")
plt.legend()
plt.savefig(f"q6_5folds_hyper_tuning.png")
