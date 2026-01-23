import numpy as np
rng = np.random.default_rng()

mnist_data = np.load("data/mnist-data.npz")
spam_data = np.load("data/spam-data.npz")

field = ("training_data", "training_labels", "test_data")

mnist_train_val_data = mnist_data[field[0]]
mnist_train_val_label = mnist_data[field[1]]
spam_train_val_data = spam_data[field[0]]
spam_train_val_label = spam_data[field[1]]

# sets aside 10,000 training images as a validation set.
mnist_tot_num = len(mnist_train_val_data)
mnist_val_num = 10000
mnist_idx = rng.permutation(np.arange(mnist_tot_num))

mnist_train_data = mnist_train_val_data[mnist_idx[mnist_val_num:]]
mnist_train_label = mnist_train_val_label[mnist_idx[mnist_val_num:]]
mnist_val_data = mnist_train_val_data[mnist_idx[:mnist_val_num]]
mnist_val_label = mnist_train_val_label[mnist_idx[:mnist_val_num]]

# sets aside 20% of the training data as a validation set.

spam_tot_num = len(spam_train_val_data)
spam_val_num = int(spam_tot_num * 0.2) # 0.2 as the rate
spam_idx = rng.permutation(np.arange(spam_tot_num))

spam_train_data = spam_train_val_data[spam_idx[spam_val_num:]]
spam_train_label = spam_train_val_label[spam_idx[spam_val_num:]]
spam_val_data = spam_train_val_data[spam_idx[:spam_val_num]]
spam_val_label = spam_train_val_label[spam_idx[:spam_val_num]]

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
