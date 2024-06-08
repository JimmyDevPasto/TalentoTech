import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
#from IPython.core.display import display, HTML
from matplotlib import animation

from sklearn.datasets import make_circles


import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

# Generación de datos
x, y = make_circles(n_samples=500, factor=0.5, noise=0.05)

# Resolución y malla para la predicción
res = 100
_x0 = np.linspace(-1.5, 1.5, res)
_x1 = np.linspace(-1.5, 1.5, res)
_Px = np.array(np.meshgrid(_x0, _x1)).T.reshape(-1, 2)
_Py = np.zeros((res, res)) + 0.5


# Visualización inicial de datos
plt.figure(figsize=(8, 8))
plt.pcolormesh(_x0, _x1, _Py, cmap="coolwarm", vmin=0, vmax=1)
plt.scatter(x[y == 0, 0], x[y == 0, 1], c="skyblue")
plt.scatter(x[y == 1, 0], x[y == 1, 1], c="salmon")
plt.tick_params(labelbottom=False, labelleft=False)
plt.show()

# Definición del modelo de red neuronal
iX = tf.placeholder('float', shape=[None, x.shape[1]])
iY = tf.placeholder('float', shape=[None])

nn = [2, 16, 8, 1]

# Capa 1
W1 = tf.Variable(tf.random_normal([nn[0], nn[1]]), name='weights_1')
b1 = tf.Variable(tf.random_normal([nn[1]]), name='bias_1')
l1 = tf.nn.relu(tf.add(tf.matmul(iX, W1), b1))

# Capa 2
W2 = tf.Variable(tf.random_normal([nn[1], nn[2]]), name='weights_2')
b2 = tf.Variable(tf.random_normal([nn[2]]), name='bias_2')
l2 = tf.nn.relu(tf.add(tf.matmul(l1, W2), b2))

# Capa 3
W3 = tf.Variable(tf.random_normal([nn[2], nn[3]]), name='weights_3')
b3 = tf.Variable(tf.random_normal([nn[3]]), name='bias_3')
py = tf.nn.sigmoid(tf.add(tf.matmul(l2, W3), b3))[:, 0]

# Función de pérdida y optimizador
loss = tf.losses.mean_squared_error(py, iY)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)
n_steps = 1000
iPY = []

# Entrenamiento del modelo
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(n_steps):
        _, _loss, _pY = sess.run([optimizer, loss, py], feed_dict={iX: x, iY: y})
        if step % 25 == 0:
            acc = np.mean(np.round(_pY) == y)
            print('Step', step, '/', n_steps, '- Loss =', _loss, ' - Acc =', acc)
        _pY = sess.run(py, feed_dict={iX: _Px}).reshape((res, res))
        iPY.append(_pY)

# # Código de animación
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.scatter(x[y == 0, 0], x[y == 0, 1], c="skyblue")
# ax.scatter(x[y == 1, 0], x[y == 1, 1], c="salmon")
# ax.tick_params(labelbottom=False, labelleft=False)

# # Inicialización del gráfico
# cax = ax.pcolormesh(_x0, _x1, iPY[0], cmap="coolwarm", vmin=0, vmax=1)

# def init():
#     cax.set_array(np.zeros((res, res)).ravel())
#     return [cax]

# def animate(fr):
#     cax.set_array(iPY[fr].ravel())
#     return [cax]

# ani = animation.FuncAnimation(fig, animate, frames=len(iPY), init_func=init, interval=50, blit=True, repeat_delay=1000)
# plt.show()

# # Mostrar la animación en HTML (si estás en un notebook Jupyter)
# HTML(ani.to_html5_video())