import numpy as np

X = np.zeros((4, 3)) # n * d
X[0] = np.array([0.2, 3.1, 1.])
X[1] = np.array([1.0, 3.0, 1.])
X[2] = np.array([-0.2, 1.2, 1.])
X[3] = np.array([1.0, 1.1, 1.])

y_train = np.array([1., 1., 0., 0.]) # 1 * n

w = np.array([-1., 1., 0.]) # 1 * d

def sigmoid(z):
    """
    :param z: np.array
        vector
    """
    return 1.0 / (1.0 + np.exp(-z))

def one_Newton(w, X, y):
    y_pred = X @ w
    p = sigmoid(y_pred) # (n,)
    
    Sigma = np.diag(p * (1 - p)) # (n, n)
    grad = X.T @ (y - p)
    H = X.T @ Sigma @ X
    
    eps = np.linalg.solve(H, grad)
    w = w + eps
    print(w)
    print(s(w, X, y))
    return w

def s(w, X, y):
    z = X @ w
    p = sigmoid(z)
    return -np.sum(
        y*np.log(p) + (1-y)*np.log(1-p)
    )

# s_0:
print(s(w, X, y_train))
# w_1, s_1
w = one_Newton(w, X, y_train)
# w_2, s_2
one_Newton(w, X, y_train)

    

