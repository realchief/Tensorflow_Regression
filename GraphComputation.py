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

# We'll use our values from [-3, 3] to create a Gaussian Distribution

sigma = 1.0
mean = 0.0
z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) / 
(2.0 * tf.pow(sigma, 2.0)))) * (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))

assert z.graph is tf.get_default_graph()
plt.plot(x.eval(), z.eval())
plt.show()