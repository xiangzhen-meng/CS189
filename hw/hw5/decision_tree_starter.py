"""
Have Fun!
- 189 Course Staff
"""
from collections import Counter

import numpy as np
from numpy import genfromtxt
import scipy.io
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from pydot import graph_from_dot_data
import io

import random
random.seed(246810)
np.random.seed(246810)

eps = 1e-5  # a small number


class DecisionTree:

    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes

    @staticmethod
    def entropy(y):
        """
        A method that takes in the labels of data stored at a node 
        and compute the entropy
        """
        if len(y) == 0:
            return 0
        value, count = np.unique(y, return_counts=True)
        if np.any(count == len(y)):
            return 0
        p = count / len(y)
        return np.sum(-p * np.log2(p))

    @staticmethod
    def information_gain(X, y, thresh):
        """
        A method that takes in some feature of the data, 
        the labels and a threshold, and compute the information gain 
        of a split using the threshold.
        X: features
        y: labels
        thresh: threshold
        """
        n = len(y)
        left = (X < thresh)
        right = (X >= thresh)
        lft_lab, rgt_lab = y[left], y[right]
        if len(lft_lab) == 0 or len(rgt_lab) == 0:
            return 0
        pnt_ent = DecisionTree.entropy(y)
        lft_w, rgt_w = len(lft_lab) / n, len(rgt_lab) / n
        lft_ent, rgt_ent = DecisionTree.entropy(lft_lab), DecisionTree.entropy(rgt_lab)
        child_ent = lft_w * lft_ent + rgt_w * rgt_ent
        return pnt_ent - child_ent

    @staticmethod
    def gini_impurity(X, y, thresh):
        # TODO
        pass

    @staticmethod
    def gini_purification(X, y, thresh):
        # TODO
        pass
    
    def should_stop(self, X, y):
        if self.max_depth == 0:
            return True
        if len(y) == 0:
            return True
        if DecisionTree.entropy(y) <= eps:
            return True
        return False

    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        if self.should_stop(X, y):
            self.data = X
            value, count = np.unique(y, return_counts=True)
            self.pred = value[np.argmax(count)]
            return self
        
        n, d = X.shape
        Best_IG = 0.0
        Best_feat = 0
        Best_thresh = 0.0
        for j in range(d):
            feature = X[:, j]
            sorted_feature = np.unique(feature)
            sorted_feature = np.sort(sorted_feature, kind='heapsort')

            for i in range(len(sorted_feature) - 1):
                thresh = (sorted_feature[i] + sorted_feature[i+1]) / 2
                IG = DecisionTree.information_gain(feature, y, thresh)
                if IG > Best_IG:
                    Best_IG = IG
                    Best_feat = j
                    Best_thresh = thresh
        # should stop implementation
        if Best_IG <= 0:
            self.data = X
            value, count = np.unique(y, return_counts=True)
            self.pred = value[np.argmax(count)]
            return self
        
        self.split_idx = Best_feat
        self.thresh = Best_thresh
        X0, y0, X1, y1 = self.split(X, y, self.split_idx, self.thresh)
        self.left = DecisionTree(max_depth=self.max_depth-1)
        self.right = DecisionTree(max_depth=self.max_depth-1)
        self.left.fit(X0, y0)
        self.right.fit(X1, y1)

    def is_leaf(self):
        return self.left is None and self.right is None

    def single_predict(self, X):
        if self.is_leaf():
            return self.pred
        if X[self.split_idx] < self.thresh:
            return self.left.single_predict(X)
        else:
            return self.right.single_predict(X)

    def predict(self, X):
        """
        predict(data): Given a data point, traverse the tree 
        to find the best label to classify the data point as. 
        Start at the root node you stored and evaluate split rules 
        at each node as you traverse until you reach a leaf node, 
        then choose that leaf node's label as your output label.
        """
        n = X.shape[0]
        y_pred = [self.single_predict(X[i, :]) for i in range(n)]
        y_pred = np.array(y_pred)
        return y_pred

    def __repr__(self):
        if self.max_depth == 0:
            return "%s (%s)" % (self.pred, self.labels.size)
        else:
            return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
                                           self.thresh, self.left.__repr__(),
                                           self.right.__repr__())


class BaggedTrees(BaseEstimator, ClassifierMixin):

    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            DecisionTreeClassifier(random_state=i, **self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        # TODO
        pass

    def predict(self, X):
        # TODO
        pass


class RandomForest(BaggedTrees):

    def __init__(self, params=None, n=200, m=1):
        if params is None:
            params = {}
        params['max_features'] = m
        self.m = m
        super().__init__(params=params, n=n)


class BoostedRandomForest(RandomForest):

    def fit(self, X, y):
        # TODO
        pass
    
    def predict(self, X):
        # TODO
        pass


def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[], fixed_onehot=None):
    # fill_mode = False

    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'
    data[data == ''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    if fixed_onehot is not None:
        # Use the exact same (col, feature) pairs from training
        onehot_map = fixed_onehot
        for col, feat in fixed_onehot:
            onehot_features.append(feat)
            onehot_encoding.append((data[:, col] == feat).astype(float))
        for col in onehot_cols:
            data[:, col] = '0'
    else:
        onehot_map = []
        for col in onehot_cols:
            counter = Counter(data[:, col])
            for term in counter.most_common():
                if term[0] == b'-1':
                    continue
                if term[-1] <= min_freq:
                    break
                onehot_features.append(term[0])
                onehot_encoding.append((data[:, col] == term[0]).astype(float))
                onehot_map.append((col, term[0]))
            data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    assert not np.any(data == '')
    data = np.hstack(
        [np.array(data, dtype=float),
         np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for j in range(data.shape[1]):
            col = data[:, j]
            missing = (col == -1)
            if not np.any(missing):
                continue
            value, count = np.unique(col[~missing], return_counts=True)
            mode = value[np.argmax(count)]
            col[missing] = mode
            
    assert not (data == -1).any()
    assert data.dtype == float
    assert not np.isnan(data).any()
    
    return data, onehot_features, onehot_map


def evaluate(clf):
    print("Cross validation", cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [
            (features[term[0]], term[1]) for term in counter.most_common()
        ]
        print("First splits", first_splits)


if __name__ == "__main__":
    dataset = "titanic"
    # dataset = "spam"
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100

    if dataset == "titanic":
        # Load titanic data
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]

        labeled_idx = np.where(y != b'')[0]
        y = np.array(y[labeled_idx], dtype=float).astype(int)
        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features, onehot_map = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8], fixed_onehot=onehot_map)
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, 1:]) + onehot_features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam_data/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features", features)
    print("Train/test size", X.shape, Z.shape)
    
    print("\n\nPart 0: constant classifier")
    print("Accuracy", 1 - np.sum(y) / y.size)

    # sklearn decision tree
    print("\n\nsklearn's decision tree")
    clf = DecisionTreeClassifier(random_state=0, **params)
    clf.fit(X, y)
    evaluate(clf)
    out = io.StringIO()
    export_graphviz(
        clf, out_file=out, feature_names=features, class_names=class_names)
    # For OSX, may need the following for dot: brew install gprof2dot
    graph = graph_from_dot_data(out.getvalue())
    graph_from_dot_data(out.getvalue())[0].write_pdf("%s-tree.pdf" % dataset)
    
    # TODO
