import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def cal_XY(mu, cov, grid_size=300, n_sig=3):
    eigval = np.linalg.eigvalsh(cov)
    max_sig = np.sqrt(eigval.max())
    
    x_lft_bound = mu[0] - n_sig * max_sig
    x_rgt_bound = mu[0] + n_sig * max_sig
    y_lft_bound = mu[1] - n_sig * max_sig
    y_rgt_bound = mu[1] + n_sig * max_sig
    X = np.linspace(x_lft_bound, x_rgt_bound, grid_size)
    Y = np.linspace(y_lft_bound, y_rgt_bound, grid_size)
    X, Y = np.meshgrid(X, Y)
    pos = np.dstack((X, Y))
    return pos, X, Y

def cal_PDF(mu, cov, pos):
    rv = multivariate_normal(mu, cov)
    Z = rv.pdf(pos)
    return Z

def gaussian_plot(X, Y, Z, save_path, level=10):
    plt.figure()
    plt.contour(X, Y, Z, levels=level)
    plt.title('Isocontours of 2D Normal Distribution')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# q6.1
mu = np.array([1, 1])
cov = np.array([[1, 0], [0, 2]])
pos, X, Y = cal_XY(mu, cov)
Z = cal_PDF(mu, cov, pos)
gaussian_plot(X, Y, Z, './q6-1.png')

# q6.2
mu = np.array([-1, 2])
cov = np.array([[2, 1], [1, 4]])
pos, X, Y = cal_XY(mu, cov)
Z = cal_PDF(mu, cov, pos)
gaussian_plot(X, Y, Z, './q6-2.png')

# q6.3
mu1 = np.array([0, 2])
mu2 = np.array([2, 0])
cov = np.array([[2, 1], [1, 1]])
pos_c, Xc, Yc = cal_XY((mu1+mu2)/2, cov)
Z1 = cal_PDF(mu1, cov, pos_c)
Z2 = cal_PDF(mu2, cov, pos_c)
Z = Z1 - Z2
gaussian_plot(Xc, Yc, Z, './q6-3.png')

# q6.4
mu1 = np.array([0, 2])
mu2 = np.array([2, 0])
cov1 = np.array([[2, 1], [1, 1]])
cov2 = np.array([[2, 1], [1, 4]])
pos_c, Xc, Yc = cal_XY((mu1+mu2)/2, cov)
Z1 = cal_PDF(mu1, cov1, pos_c)
Z2 = cal_PDF(mu2, cov2, pos_c)
Z = Z1 - Z2
gaussian_plot(Xc, Yc, Z, './q6-4.png')

# q6.5
mu1 = np.array([1, 1])
mu2 = np.array([-1, -1])
cov1 = np.array([[2, 0], [0, 1]])
cov2 = np.array([[2, 1], [1, 2]])
pos_c, Xc, Yc = cal_XY((mu1+mu2)/2, cov)
Z1 = cal_PDF(mu1, cov1, pos_c)
Z2 = cal_PDF(mu2, cov2, pos_c)
Z = Z1 - Z2
gaussian_plot(Xc, Yc, Z, './q6-5.png')
