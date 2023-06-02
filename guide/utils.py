import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_boundary(ax, x, y, classifier, step=0.02):
    markers = ['s', 'x', 'o', '^', 'v']
    colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan']
    cmap = ListedColormap(colors=colors[:len(np.unique(y))])
    x1_min, x1_max = x[:, 0].min()-1, x[:, 0].max()+1
    x2_min, x2_max = x[:, 1].min()-1, x[:, 1].max()+1
    x1_coordinates, x2_coordinates = np.meshgrid(np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step))
    Z = classifier.predict(np.array([x1_coordinates.ravel(), x2_coordinates.ravel()]).T)
    Z = Z.reshape(x1_coordinates.shape)
    ax.contourf(x1_coordinates, x2_coordinates, Z, alpha=0.4, cmap=cmap)
    ax.set_xlim(x1_coordinates.min(), x1_coordinates.max())
    ax.set_ylim(x2_coordinates.min(), x2_coordinates.max())
    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=x[y == cl, 0], y=x[y == cl, 1], alpha=0.6, c=colors[idx], marker=markers[idx], label=cl)

def visualize_decision_boundary(x_train, y_train, x_test, y_test, classifier, xlabel="", ylabel=""):
    fig = plt.figure(figsize=(10, 4), layout='tight')
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    ax1 = fig.add_subplot(1, 2, 1)
    plot_decision_boundary(ax1, x_train, y_train, classifier=classifier)
    plt.title("train", loc='center')
    plt.legend(loc="best")  
    ax2 = fig.add_subplot(1, 2, 2)
    plot_decision_boundary(ax2, x_test, y_test, classifier=classifier)
    plt.title("test", loc='center')
    plt.legend(loc="best")
    plt.show()