
import numpy as np
import matplotlib.pyplot as plt

num_features = 2
num_iter = 2000
learning_rate = 0.01

x = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32) # input
# y = np.array([0, 0, 1, 0], np.float32) # AND
# y = np.array([0, 1, 1, 1], np.float32) # OR
y = np.array([0, 1, 0, 1], np.float32) # XOR

w = np.zeros(num_features, np.float32) # initialize weights as zeroes
w0 = np.zeros(1, np.float32) # initalize w0 as zero

m, n = np.shape(x) # number of samples m and number of features n

for k in range(num_iter):
    for j in range(m):
      g_z = x[j, :].dot(w) + w0  # sum of weighted inputs z
      h = 1.0 / (1.0 + np.exp(-g_z)) # hypothesis is sigmoid g(z)

      err = y[j] - h # error is expected output - hypothesis
      delta_w = err * x[j, :]
      delta_b = err
      w = w + learning_rate * delta_w  # update weights
      w0 = w0 + learning_rate * delta_b # update w0

# plot the fitted line; we only need two points to plot the line
plot_x = np.array([np.min(x[:, 0] - 0.2), np.max(x[:, 1] + 0.2)])
plot_y = - 1 / w[1] * (w[0] * plot_x + w0) # w0*x + w1*y + b = 0 --> y = (-1/w1) (w0*x + b)

print('W:' + str(w))
print('b:' + str(b))
print('plot_y: ' + str(plot_y))

plt.scatter(x[:, 0], x[:, 1], c=y, s=100, cmap='viridis')
plt.plot(plot_x, plot_y, color='k', linewidth=2)
plt.xlim([-0.2, 1.2]);
plt.ylim([-0.2, 1.25]);
plt.show()