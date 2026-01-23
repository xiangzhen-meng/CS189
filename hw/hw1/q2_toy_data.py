import matplotlib.pyplot as plt
import numpy as np

# load data
toy_data = np.load("data/toy-data.npz")
field = ("training_data", "training_labels", "test_data")
train_dat = toy_data[field[0]]
train_lab = toy_data[field[1]]

plt.scatter(train_dat[:, 0], train_dat[:, 1], c = train_lab)

# Plot the decision boundary
x = np.linspace(-5, 5, 100)
w = np.array([-0.4528, -0.5190])
b = 0.1471
y = -(w[0] * x + b) / w[1]
plt.plot(x, y, 'k')

# Plot the margins
upper_b = b + 1
lower_b = b - 1
upper_margin = -(w[0] * x + upper_b) / w[1]
lower_margin = -(w[0] * x + lower_b) / w[1]
plt.plot(x, upper_margin, 'k', c = 'blue')
plt.plot(x, lower_margin, 'k', c = 'red')

plt.savefig("q1-plot-toy-data.png")
plt.show()