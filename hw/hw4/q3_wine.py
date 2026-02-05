import numpy as np
import scipy
import matplotlib.pyplot as plt
rng = np.random.default_rng(seed=189)

# ***************************** #
# Load Origin Data & Preprocess #
# ***************************** #

origin = scipy.io.loadmat("data.mat")
# data: (5000, 12)
# label: (5000)
# data_test: (1000, 12)
ori_data, ori_label, data_test = origin['X'], origin['y'].ravel(), origin['X_test']
assert len(ori_data) == len(ori_label)
tot_num = len(ori_data)

# preprocess: Standardization
ori_data_mean = np.mean(ori_data, axis=0)
ori_data_std = np.std(ori_data, axis=0, ddof=0)
ori_data = (ori_data - ori_data_mean) / ori_data_std
data_test = (data_test - ori_data_mean) / ori_data_std

# add fictitious dimension
ori_data = np.hstack([ori_data, np.ones((ori_data.shape[0], 1))])  # keep ori_data as before
data_test = np.hstack([data_test, np.ones((data_test.shape[0], 1))])  # data_test may have different number of rows; use its own shape

# k-fold Cross Validation
indices = rng.permutation(tot_num)
k_fold = 5
split_idx = np.array_split(indices, k_fold)
fold_data = [ori_data[idx] for idx in split_idx]
fold_label = [ori_label[idx] for idx in split_idx]
def current_fold(X, cur_idx):
    """
    :params X: np.array
        fold_data or fold_label
    :params cur_idx: int
        current i of k folds
        
    return: 
        current training set: np.array
        current validation set: np.array
    """
    train_set = np.concatenate(X[:cur_idx] + X[cur_idx + 1:], axis=0)
    return train_set, X[cur_idx]

# **************************************************** #
# Batch Gradient Descent & Stochastic Gradient Descent #
# **************************************************** #

# SUMMARY:
# for BatchGD: learning rate = 0.0027825594022071257, lambda = 0.021544346900318832
# for SGD: learning rate = 0.0005994842503189409, lambda = 1e-10
# for Decay-SGD: delta = 483.2930238571752, lambda = 0.007742636826811269

# BatchGD ACC = 0.995599, max_iter = 1000
# SGD ACC = 0.992800, max_iter = 50
# Decay SGD ACC = 0.9904, max_iter = 50

# model init
def sigmoid(X):
    return scipy.special.expit(X)

class LogisticRegression():
    def __init__(self, lr=1e-4, lam=0.007742, delta=100, max_iter=50):
        self.lr = lr
        self.lam = lam
        self.delta = delta
        self.max_iter = max_iter
        self.w = None
        self.loss_history = []

    def loss(self, X, y):
        """
        :param X: (n, d)
        :param y: (n,)
        :param w: (d,)
        J(w) = - [y^T * log(s) + (1 - y)^T * log(1 - s)] + lambda * ||w||^2
        """
        s = sigmoid(X @ self.w)
        s = np.clip(s, 1e-12, 1-1e-12)
        loss = np.sum(-(y * np.log(s) + (1-y) * np.log(1-s)))
        reg = self.lam * np.sum(self.w[:-1] ** 2)
        return loss + reg
    
    def gradient(self, X, y):
        """
        :param X: (n, d)
        :param y: (n,)
        :param w: (d,)
        nabla J(w) = 2 * lambda * w + X^T * (s - y)
        """
        s = sigmoid(X @ self.w)
        s = np.clip(s, 1e-12, 1-1e-12)
        grad = X.T @ (s - y)
        reg_grad = np.zeros_like(self.w)
        reg_grad[:-1] = 2 * self.lam * self.w[:-1]
        return grad + reg_grad

    def fit(self, X, y, batch_size=None, Decay=False):
        n, d = X.shape
        self.w = np.zeros(d)
        if batch_size is None:
            batch_size = n
        t = 1
        for epoch in range(self.max_iter):
            perm = rng.permutation(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                # loss = self.loss(X[idx], y[idx])
                # self.loss_history.append(loss)
                grad = self.gradient(X[idx], y[idx])
                if Decay:
                    lr = self.delta / t
                else:
                    lr = self.lr
                self.w -= lr * grad
                t += 1
            
    def predict_proba(self, X):
        s = sigmoid(X @ self.w)
        s = np.clip(s, 1e-12, 1-1e-12)
        return s

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# ******************* #
# Tuning Hyper-Params #
# ******************* #

# model pre-tuning: learning rate
""" lr_list = [1e-8, 1e-6, 1e-4, 5 * (1e-4), 1e-3, 5 * (1e-3), 1e-2, 1e-1]

training_loss_lst = []
for lr in lr_list:
    model = LogisticRegression(lr=lr, lam=5 * (1e-3))
    model.fit(ori_data, ori_label, batch_size=1)
    training_loss_lst.append(min(model.loss_history))
training_loss_lst = np.array(training_loss_lst)
print(np.argmin(training_loss_lst))
print(np.min(training_loss_lst))

fig, ax = plt.subplots()
ax.plot(lr_list, training_loss_lst)
ax.set_xlabel("learning rate")
ax.set_xscale("log")
ax.set_ylabel("training loss")
plt.savefig("./pics/q3-pretune-SGD-lr.png")
plt.show() """

# model tuning: lambda
# lam_lst = [1e-10, 1e-5, 1e-4, 1e-3, 5*(1e-3), 1e-2, 5*(1e-2), 1e-1]
""" lam_lst = np.logspace(-3, -1, num=10) 
acc_lst = []
for lam in lam_lst:
    kth_acc = 0.0
    for k in range(k_fold):
        cur_dat, cur_dat_vali = current_fold(fold_data, k)
        cur_lab, cur_lab_vali = current_fold(fold_label, k)
        model = LogisticRegression(lam=lam)
        model.fit(cur_dat, cur_lab, batch_size=1, Decay=True)
        prediction = model.predict(cur_dat_vali)
        kth_acc += np.mean([prediction == cur_lab_vali])
    kth_acc /= k_fold
    acc_lst.append(kth_acc)
acc_lst = np.array(acc_lst)
print(np.argmax(acc_lst))
print(np.max(acc_lst))
print(lam_lst[np.argmax(acc_lst)])
fig, ax = plt.subplots()
ax.plot(lam_lst, acc_lst)
ax.set_xlabel("lambda")
ax.set_xscale("log")
ax.set_ylabel("Cross-Validation loss")
plt.savefig("./pics/q3-CV-SGD-Decay-lambda.png")
plt.show()  """

# model tuning: learning rate
# lr_lst = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1, 5, 7]
""" lr_lst = np.logspace(-5, -3, num=10) 

acc_lst = []
for lr in lr_lst:
    kth_acc = 0.0
    for k in range(k_fold):
        cur_dat, cur_dat_vali = current_fold(fold_data, k)
        cur_lab, cur_lab_vali = current_fold(fold_label, k)
        model = LogisticRegression(lr=lr)
        model.fit(cur_dat, cur_lab, batch_size=1)
        prediction = model.predict(cur_dat_vali)
        kth_acc += np.mean([prediction == cur_lab_vali])
    kth_acc /= k_fold
    acc_lst.append(kth_acc)
acc_lst = np.array(acc_lst)
print(np.argmax(acc_lst))
print(max(acc_lst))
print(lr_lst[np.argmax(acc_lst)])
fig, ax = plt.subplots()
ax.plot(lr_lst, acc_lst)
ax.set_xlabel("learning rate")
ax.set_xscale("log")
ax.set_ylabel("Cross-Validation Accuracy")
plt.savefig("./pics/q3-CV-lr-SGD-precise.png")
plt.show()   """

# model tuning: delta
# dt_lst = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
""" dt_lst = np.logspace(1, 3, num=20) 

acc_lst = []
for dt in dt_lst:
    kth_acc = 0.0
    for k in range(k_fold):
        cur_dat, cur_dat_vali = current_fold(fold_data, k)
        cur_lab, cur_lab_vali = current_fold(fold_label, k)
        model = LogisticRegression(delta=dt)
        model.fit(cur_dat, cur_lab, batch_size=1, Decay=True)
        prediction = model.predict(cur_dat_vali)
        kth_acc += np.mean([prediction == cur_lab_vali])
    kth_acc /= k_fold
    acc_lst.append(kth_acc)
acc_lst = np.array(acc_lst)
print(np.argmax(acc_lst))
print(max(acc_lst))
print(dt_lst[np.argmax(acc_lst)])
fig, ax = plt.subplots()
ax.plot(dt_lst, acc_lst)
ax.set_xlabel("delta")
ax.set_xscale("log")
ax.set_ylabel("Cross-Validation Accuracy")
plt.savefig("./pics/q3-CV-Decay-SGD-dt-precise.png")
plt.show()  """ 

# model tuning: max_iteration times
# iter_times = [200, 500, 700, 1000, 2000, 5000]
""" iter_times = np.linspace(200, 1000, num=20, dtype=np.int32)
acc_lst = []
for it in iter_times:
    kth_acc = 0.0
    for k in range(k_fold):
        cur_dat, cur_dat_vali = current_fold(fold_data, k)
        cur_lab, cur_lab_vali = current_fold(fold_label, k)
        model = LogisticRegression(max_iter=it)
        model.fit(cur_dat, cur_lab)
        prediction = model.predict(cur_dat_vali)
        kth_acc += np.mean([prediction == cur_lab_vali])
    kth_acc /= k_fold
    acc_lst.append(kth_acc)
acc_lst = np.array(acc_lst)
print(np.argmax(acc_lst))
print(max(acc_lst))
print(iter_times[np.argmax(acc_lst)])
fig, ax = plt.subplots()
ax.plot(iter_times, acc_lst)
ax.set_xlabel("iteration times")
ax.set_ylabel("Cross-Validation Accuracy")
plt.savefig("./pics/q3-CV-iter.png")
plt.show()  """

# ************* #
# Kaggle Submit #
# ************* #

model = LogisticRegression(lr=0.002783, lam=0.021544, max_iter=1000)
model.fit(ori_data, ori_label)
y_pred = model.predict(data_test)
# save the results into a kaggle accepted csv
import pandas as pd
def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1 # Ensures that the index starts at 1
    df.to_csv('submission_wine.csv', index_label='Id')

results_to_csv(y_pred)
