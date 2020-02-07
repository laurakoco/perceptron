
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

NUM_FEATURES = 2
NUM_ITER = 2000
learning_rate = 0.01

tf.compat.v1.disable_eager_execution() # had to add this due to eager execution error 

x = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)  # 4x2, input
# y = np.array([0, 0, 1, 0], np.float32)  # 4, correct output, AND operation
y = np.array([0, 1, 1, 1], np.float32)  # OR operation

y_for_plot = y # laura added this line
y = np.reshape(y, [4, 1])  # convert to 4x1

X = tf.compat.v1.placeholder(tf.float32, shape=[4, 2])
Y = tf.compat.v1.placeholder(tf.float32, shape=[4, 1])

W = tf.Variable(tf.zeros([NUM_FEATURES, 1]), tf.float32)
B = tf.Variable(tf.zeros([1, 1]), tf.float32)

yHat = tf.sigmoid(tf.add(tf.matmul(X, W), B))  # 4x1
err = Y - yHat
deltaW = tf.matmul(tf.transpose(a=X), err)  # have to be 2x1
deltaB = tf.reduce_sum(input_tensor=err, axis=0)  # 4, have to 1x1. sum all the biases? yes
W_ = W + learning_rate * deltaW
B_ = B + learning_rate * deltaB

step = tf.group(W.assign(W_), B.assign(B_))  # to update the values of weights and biases.

sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

for k in range(NUM_ITER):
  sess.run([step], feed_dict={X: x, Y: y})

W = np.squeeze(sess.run(W))
b = np.squeeze(sess.run(B))

# Now plot the fitted line. We need only two points to plot the line
plot_x = np.array([np.min(x[:, 0] - 0.2), np.max(x[:, 1] + 0.2)])
plot_y = -1 / W[1] * (W[0] * plot_x + b)
plot_y = np.reshape(plot_y, [2, -1])
plot_y = np.squeeze(plot_y)

print('W: ' + str(W))
print('b: ' + str(b))
print('plot_y: ' + str(plot_y))

plt.scatter(x[:, 0], x[:, 1], c=y_for_plot, s=100, cmap='viridis') # laura changed this line
plt.plot(plot_x, plot_y, color='k', linewidth=2)
plt.xlim([-0.2, 1.2]);
plt.ylim([-0.2, 1.25]);
plt.show()