import sklearn as sk 
import sklearn.neural_network 
from IPython.core.display import display, HTML


# Generación de datos
x, y = make_circles(n_samples=500, factor=0.5, noise=0.05)

lr= 0.01 #learning rate
nn= [2,16,8,1] #numero de neuronas por capa 

# creamos el objeto del modelo de red neuronal multicapa. 
clf= sk.neural_network.MLPRegressor(solver='sgd',learning_rate_init=lr,hidden_layer_sizes=tuple(nn[1:]),verbose=True, n_iter_no_chage=1000, batch_size=64)

# Y los entrenamos con nuestro datos. 

clf.fit(X,Y) 

