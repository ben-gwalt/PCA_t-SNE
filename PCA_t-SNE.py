import os
import gzip
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
RS = 123

os.chdir('C:/Users/yours/PycharmProjects/PCA_t-SNE')

# Fashion MNIST reader


def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


X_train, y_train = load_mnist('fashion-mnist-master/data/fashion', kind='train')


# function to visualize outputs

def fashion_scatter(x, colors):

    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # creating scatter plot
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # adding the labels for each digit
    txts = []

    for i in range(num_classes):

        # position of each label at median of data points

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


# taking subset of first 20k for visualization

x_subset = X_train[0:200000]
y_subset = y_train[0:200000]

# performing and visualizing PCA

time_start = time.time()

pca = PCA(n_components=4)
pca_result = pca.fit_transform(x_subset)

print ('PCA done! Time elapsed: {} seconds'.format(time.time()-time_start))

# storing components in new df and checking variance explained

pca_df = pd.DataFrame(columns = ['pca1', 'pca2', 'pca3', 'pca4'])

pca_df['pca1'] = pca_result[:,0]
pca_df['pca2'] = pca_result[:,1]
pca_df['pca3'] = pca_result[:,2]
pca_df['pca4'] = pca_result[:,3]

print('Variance Explained per component: {}'.format(pca.explained_variance_ratio_))

# the first and second component explain nearly 48% of the variance and
# they will be used for visualization

top_two_comp = pca_df[['pca1', 'pca2']]
fashion_scatter(top_two_comp.values, y_subset)

# performing and visualizing t-SNA

time_start = time.time()

fashion_TSNE = TSNE(random_state=RS).fit_transform(x_subset)

print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

fashion_scatter(fashion_TSNE, y_subset)

# using PCA in combination with t-SNA
# reducing dimensions before feeding into t-SNA

time_start = time.time()

pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(x_subset)

print('PCA with 50 components done! Time elapsed: {} seconds'.format(time.time()-time_start))

print('Cumulative variance explained by 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

# feeding into t-SNA

time_start = time.time()

fashion_pca_tsne = TSNE(random_state=RS).fit_transform(pca_result_50)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

fashion_scatter(fashion_pca_tsne, y_subset)








