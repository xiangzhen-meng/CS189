import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
rng = np.random.default_rng(seed=42)

# ******* Data Preprocess *******

origin = np.load("data/mnist-data-hw3.npz")
original_data, original_label = origin["training_data"], origin["training_labels"]
test_data = origin["test_data"]

def L2_normalization(data):
    """
    return the L2-normalized data: np.array
    :data: np.array
        resource data
    """
    # flatten
    if data.ndim > 2:
        data = data.reshape(data.shape[0], -1)
    L2_norms = np.linalg.norm(data, axis=1, keepdims=True) + 1e-15
    return data / L2_norms

def split_train_vali(data, label, vali_num=10000):
    """
    return in order:
        data train
        data vali
        label train
        label vali
    :data: np.array
        original data
    :label: np.array
        original label
    :vali_num: int
        size of validation set, 10000 in defalt case
    """
    length = len(data)
    idx = rng.permutation(length)
    return data[idx[vali_num:]], data[idx[:vali_num]], label[idx[vali_num:]], label[idx[:vali_num]]

original_data = L2_normalization(original_data)
data_train, data_vali, label_train, label_vali = split_train_vali(original_data, original_label)

digits = [i for i in range(10)]
def estimate_gaussian_params(data, label):
    """
    :param data: np.array
    :param label: np.array
    
    return 3 params: np.array, list, list
        mean for each class 
        cov matrix for each class
        class_sizes of each class
    """
    means = []
    covs = []
    class_sizes = []
    for i in digits:
        data_cur = data[label == i]
        size_cur = len(data_cur)
        if size_cur == 0:
            # no samples for this class: use zeros to avoid NaNs downstream
            mu_cur = np.zeros(data.shape[1])
            cov_cur = np.zeros((data.shape[1], data.shape[1]))
        else:
            mu_cur = np.mean(data_cur, axis=0)
            cov_cur = np.cov(data_cur, rowvar=False)
        means.append(mu_cur)
        covs.append(cov_cur)
        class_sizes.append(size_cur)

    means = np.array(means)
    return means, covs, class_sizes

def calculate_pooled_cov(covs, sizes):
    """
    :param covs: list(np.array)
        cov of each class
    :param sizes: list(int)
        size of each class
        
    return LDA_cov: np.array
        the mutual cov of all classes
    """
    n_tot = sum(sizes)
    LDA_cov = sum(covs[i] * sizes[i] for i in digits) / n_tot
    return LDA_cov

# ******* Classifier Classes *******

class Gaussian_Classifier:
    def __init__(self):
        self.means = None
        self.covs = None
        self.prior_prob = None
        
    def fit(self, data, label):
        self.means, self.covs, sizes = estimate_gaussian_params(data, label)
        n_tot = sum(sizes)
        self.prior_prob = np.array([sz/n_tot for sz in sizes])
        
    def predict(self, data):
        raise NotImplementedError


class LDA(Gaussian_Classifier):
    def __init__(self):
        super().__init__()
        self.pooled_cov = None
        
    def fit(self, data, label):
        super().fit(data, label)
        sizes = np.bincount(label)
        self.pooled_cov = calculate_pooled_cov(self.covs, sizes)
        
    def predict(self, X):
        scores = np.zeros((len(X), len(digits)))
        # number of data, number of digits(classes)
        cov = self.pooled_cov
        eps = 1e-4
        W = np.linalg.solve(cov + eps * np.eye(cov.shape[0]), self.means.T).T
        b = -0.5 * np.sum(self.means * W, axis=1) + np.log(self.prior_prob)
        scores = X @ W.T + b   # (N, K)
        y_pred = np.argmax(scores, axis=1)
        return y_pred

class QDA(Gaussian_Classifier):
    def __init__(self):
        super().__init__()

    def predict(self, X):
        scores = np.zeros((len(X), len(digits)))
        # N number of data, K number of digits(classes)
        eps = 1e-4
        for k in digits:
            mu = self.means[k]
            cov = self.covs[k]
            D = cov.shape[0]
            cov = cov + eps * np.eye(D)
            sgn, log_det = np.linalg.slogdet(cov)
            assert sgn > 0
            X_centered = X - mu
            sol = np.linalg.solve(cov, X_centered.T).T
            quad = np.sum(X_centered * sol, axis=1)
            scores[:, k] = (-0.5 * quad - 0.5 * log_det + np.log(self.prior_prob[k]))
        y_pred = np.argmax(scores, axis=1)
        return y_pred

# ******* Validation *******

def evaluation_metric(true_lab, pred_lab):
    '''
    >>> true_lab = np.array([1, 2, 3, 4])
    >>> pred_lab = np.array([1, 1, 3, 4])
    >>> evaluation_metric(true_lab, pred_lab)
    0.75
    '''
    assert len(true_lab) == len(pred_lab)
    tot_len = len(true_lab)
    same_cnt = np.sum(true_lab == pred_lab)
    return same_cnt / tot_len

# LDA Test
training_pts = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, len(data_train)]
vali_rate = []
for pts in training_pts:
    cur_data = data_train[:pts]
    cur_label = label_train[:pts]
    LDA_clf = LDA()
    LDA_clf.fit(cur_data, cur_label)
    y_pred = LDA_clf.predict(data_vali)
    y_vali = label_vali
    vali_rate += [evaluation_metric(y_vali, y_pred)]

vali_rate = np.array(vali_rate)
print("max_LDA_ACC_Rate", np.max(vali_rate), np.argmax(vali_rate))
    
# plotting
fig, ax = plt.subplots()
ax.plot(training_pts, vali_rate, marker='o')
ax.set_title("LDA Validation Accuracy vs Training Size")
ax.set_xlabel("number of training examples")
ax.set_ylabel("validation accuracy")
plt.savefig("./pics/q8-3a.png")
plt.show()

""" # QDA Test
vali_rate = []
for pts in training_pts:
    cur_data = data_train[:pts]
    cur_label = label_train[:pts]
    QDA_clf = QDA()
    QDA_clf.fit(cur_data, cur_label)
    y_pred = QDA_clf.predict(data_vali)
    y_vali = label_vali
    vali_rate += [evaluation_metric(y_vali, y_pred)]
    
vali_rate = np.array(vali_rate)
print("max_QDA_ACC_Rate", np.max(vali_rate), np.argmax(vali_rate))    

# plotting
fig, ax = plt.subplots()
ax.plot(training_pts, vali_rate, marker='o')
ax.set_title("QDA Validation Accuracy vs Training Size")
ax.set_xlabel("number of training examples")
ax.set_ylabel("validation accuracy")
plt.savefig("./pics/q8-3b.png")
plt.show() """

# ******* Test *******
test_data = L2_normalization(test_data)
cur_data = data_train[:50000]
cur_label = label_train[:50000]
LDA_clf = LDA()
LDA_clf.fit(cur_data, cur_label)
submit_pred = LDA_clf.predict(test_data)

from scripts.save_csv import results_to_csv
results_to_csv(submit_pred)