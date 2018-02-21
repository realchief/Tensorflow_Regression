import tensorflow as tf
import matplotlib.pyplot as plt

n_values = 32
x = tf.linspace(-3.0, 3.0, n_values)

sess = tf.Session()
result = sess.run(x)
x.eval(session=sess)

sess.close()
sess = tf.InteractiveSession()
x.eval()

# create a Gaussian Distribution with values from [-3, 3]
sigma = 1.0
mean = 0.0
z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) / 
(2.0 * tf.pow(sigma, 2.0)))) * (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))

assert z.graph is tf.get_default_graph()
plt.plot(x.eval(), z.eval())
plt.show()

# We can find out the shape of a tensor 
print(z.get_shape())
print(z.get_shape().as_list())
print(tf.shape(z).eval())

# Multiply the two to get a 2d gaussian
z_2d = tf.matmul(tf.reshape(z, [n_values, 1]), tf.reshape(z, [1, n_values]))
plt.imshow(z_2d.eval())
plt.show()

# For fun let's create a gabor patch:
x = tf.reshape(tf.sin(tf.linspace(-3.0, 3.0, n_values)), [n_values, 1])
y = tf.reshape(tf.ones_like(x), [1, n_values])
z = tf.multiply(tf.matmul(x, y), z_2d)
plt.imshow(z.eval())
plt.show()