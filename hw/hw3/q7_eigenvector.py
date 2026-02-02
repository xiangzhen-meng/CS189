import numpy as np
import matplotlib
import matplotlib.pyplot as plt

seed = 2026
rng = np.random.default_rng(seed)

def generate_sample(rng, n):
    """
    rng: random number generator
    returns a single random sample pt of X1, X2
    """
    x1 = rng.normal(loc=3.0, scale=9.0, size=n)
    x2_noise = rng.normal(loc=4.0, scale=2.0, size=n)
    x2 = x1 * 0.5 + x2_noise
    ret = np.vstack((x1, x2))
    return ret

sample = generate_sample(rng, 100)

# q7.1
sample_mean = sample.mean(axis=1)
print("mean = ")
print(sample_mean)
# q7.2
sample_cov = np.cov(sample)
print("covariance matrix = ")
print(sample_cov)
# q7.3
eigval, eigvec = np.linalg.eigh(sample_cov)
print("eigenvectors")
print(eigvec)
print("eigenvalues")
print(eigval)

# q7.4
fig, ax = plt.subplots()
x = sample[0]
y = sample[1]
ax.scatter(x, y)

for i in range(len(eigvec)):
    direction = eigvec[:, i]
    length = eigval[i]
    ax.quiver(sample_mean[0], sample_mean[1], 
              length * direction[0], length * direction[1], 
              angles='xy', scale_units='xy', scale=1, width=0.01)

ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_aspect('equal', adjustable='box')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Samples and Covariance Eigenvectors')
ax.grid(True)

plt.savefig('./q7-4.png', dpi=300, bbox_inches='tight')
plt.show()

# q7.5 centering
sample_centered = sample - sample.mean(axis=1, keepdims=True)
x = sample_centered[0]
y = sample_centered[1]

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Centered')
ax.grid(True)
plt.savefig('./q7-5-centered.png', dpi=300, bbox_inches='tight')
plt.show()

# q7.5 decorrelating
sample_decorrelated = eigvec.T @ sample_centered
x = sample_decorrelated[0]
y = sample_decorrelated[1]

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Decorrelated')
ax.grid(True)
plt.savefig('./q7-5-decorrelated.png', dpi=300, bbox_inches='tight')
plt.show()

# q7.5 sphering
sphering_mat = np.diag(1.0 / np.sqrt(eigval))
sample_sphered = sphering_mat @ sample_decorrelated
x = sample_sphered[0]
y = sample_sphered[1]
fig, ax = plt.subplots()
ax.scatter(x, y)
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Sphered')
ax.grid(True)
plt.savefig('./q7-5-sphered.png', dpi=300, bbox_inches='tight')
plt.show()