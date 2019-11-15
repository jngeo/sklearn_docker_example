import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn import decomposition
from sklearn import datasets

np.random.seed(5)


centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()
print('Loading iris dataset...')

X = iris.data
y = iris.target
print('shape of X: ', np.shape(X))
print('shape of y: ', np.shape(y))


### apply pca
n_components=3
pca = decomposition.PCA(n_components=n_components)
pca.fit(X)
print('Fitting pca...')
print(pca)

X_pca = pca.transform(X)

print('shape of X_pca: ', np.shape(X_pca))
print('number of components: ', n_components)
print('pca total explained variance: ', pca.explained_variance_ratio_.cumsum())


'''
fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
           edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()

'''