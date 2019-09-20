import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn import neighbors


def knn_comparison(data, n_neighbors=1):
    '''
    this function find k-NN and plots dat
    '''

    x = data[:, :2]
    y = data[:, 2]

    # grid cell size
    h = 0.02
    cmap_light = ListedColormap(['#FFAAAA',  '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    # the core classifier : k-NN
    k_nn_model = neighbors.KNeighborsClassifier(n_neighbors)
    k_nn_model.fit(x, y)
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    # creat mesh grid (x_min, y_min) to (x_max, y_max) with grid space h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # predict the value (either 0 or 1) of each element in grid

    # xx.ravel() will give a flatten array

    # np.c_ : Translates slice objects to concatenation along the second axis.
    # > np.c_[np.array([1,2,3]), np.array([4,5,6])]
    # > array([[1, 4],
    #          [2, 5],
    #          [3, 6]])   (source: np.c_ documentation)

    z = k_nn_model.predict(np.c_[xx.ravel(), yy.ravel()])

    # convert the out back to the xx shape (we need it to plot the decission boundry)
    z = z.reshape(xx.shape)

    # pcolormesh will plot the (xx,yy) grid with colors according to the values of Z
    # it looks like decision boundry
    plt.figure()
    plt.pcolormesh(xx, yy, z, cmap=cmap_light)

    # scatter plot of with given points
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold)

    # defining scale on both axises
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # set the title
    plt.title("k value = " + str(n_neighbors))

    plt.show()


data = np.genfromtxt("data/2.concerticcir1.csv", delimiter=',')
knn_comparison(data, 1)
knn_comparison(data, 5)
knn_comparison(data, 15)
knn_comparison(data, 30)
#%%
data = np.genfromtxt("data/3.concertriccir2.csv", delimiter=',')
knn_comparison(data, 1)
knn_comparison(data, 5)
knn_comparison(data, 15)
#%%
data = np.genfromtxt("data/4.linearsep.csv", delimiter=',')
knn_comparison(data, 1)
knn_comparison(data, 5)
knn_comparison(data)
#%%
data = np.genfromtxt("data/5.outlier.csv", delimiter=',')
knn_comparison(data, 1)
knn_comparison(data, 5)
knn_comparison(data)
#%%
data = np.genfromtxt("data/7.xor.csv", delimiter=',')
knn_comparison(data, 1)
knn_comparison(data, 5)
knn_comparison(data)
#%%
data = np.genfromtxt("data/8.twospirals.csv", delimiter=',')
knn_comparison(data, 1)
knn_comparison(data, 5)
knn_comparison(data)
#%%
data = np.genfromtxt("data/9.random.csv", delimiter=',')
knn_comparison(data, 1)
knn_comparison(data, 5)
knn_comparison(data)



