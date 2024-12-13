import numpy as np
from matplotlib import pyplot as plt

# %% TP1 et 2
def plotGallery(images, n=16, title=None):
    # Affiche les n premières images contenues dans images
    # images est de taille Nb image*Ny*Nx
    n = min(n, images.shape[0])
    nSubplots = int(np.ceil(np.sqrt(n)))
    fig, axs = plt.subplots(nSubplots, nSubplots)
    for i in range(n):
        axs[i // nSubplots, i % nSubplots].imshow(images[i], cmap=plt.cm.gray)
        axs[i // nSubplots, i % nSubplots].set_xticks([])
        axs[i // nSubplots, i % nSubplots].set_yticks([])
    if title:
        plt.suptitle(title)

def plotHistoClasses(lbls):
    nLbls = np.array([[i, np.where(lbls == i)[0].shape[0]] for i in np.unique(lbls)])
    plt.figure()
    plt.bar(nLbls[:, 0], nLbls[:, 1])
    plt.title("Nombre d'exemples par classe")
    plt.grid(axis='y')

def plotPerf(tReco, sLbl=None):
    n, nLbl = tReco.shape[1], len(sLbl)
    x = np.arange(1, n+1)

    plt.figure()
    for i, tR in enumerate(tReco):
        kR = np.argmax(tR)
        plt.plot(x, tR, label=sLbl[i] if i < nLbl else "")
        plt.scatter(kR+1, tR[kR], 100, "C3", marker="o")
        plt.plot([0, kR+1, kR+1], [tR[kR], tR[kR], 0], "--k")
    plt.gca().set(xlim=[0, n], ylim=[np.min(tReco)*0.9, np.max(tReco)*1.1])
    plt.grid()
    plt.title("Taux de reconnaissance")
    plt.legend()


# %% TP5
def aff_donnees(X,y,bornex,borney,s):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=s, cmap='winter')
    plt.xlim(bornex)
    plt.ylim(borney)



def visualize_classifier(model, X, y, ax = None):
    if not ax:
        plt.figure( )
        ax = plt.gca( ) # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c = y, s = 30, cmap = 'rainbow', clim = (y.min(), y.max()), zorder = 3)
    ax.axis('tight')
    ax.axis('off')
    xlim, ylim = ax.get_xlim( ), ax.get_ylim( )
    xx, yy = np.meshgrid(np.linspace(*xlim, num = 200), np.linspace(*ylim, num = 200))
    Z = model.predict(np.c_[xx.ravel( ), yy.ravel( )]).reshape(xx.shape) 
    # Create a color plot with the results
    nClasses = len(np.unique(y))
    ax.contourf(xx, yy, Z, alpha = 0.3, levels = np.arange(nClasses + 1) - 0.5, cmap = 'rainbow', zorder = 1)
    ax.set(xlim = xlim, ylim = ylim)

# %% TP6

    return ax


def aff_frontiere(X,y,bornex,borney,model):
    aff_donnees(X,y,bornex,borney,50)
    xx, yy = np.meshgrid(np.linspace(bornex[0], bornex[1],50), np.linspace(borney[0], borney[1],50))
    xy = np.concatenate((np.reshape(xx,(xx.shape[0]*xx.shape[1],1)),np.reshape(yy,(yy.shape[0]*yy.shape[1],1))),axis=1)
    P = model.predict(xy)
    aff_donnees(xy,P,bornex,borney,1) 


# %% TP7
def affiche(history, fig=None, axs=None):
    if not fig:
        fig, axs = plt.subplots(1, 2)
    # summarize history for accuracy
    axs[0].plot(history.history['accuracy'])
    axs[0].plot(history.history['val_accuracy'])
    axs[0].set_title('model accuracy')
    axs[0].set_ylabel('accuracy')
    axs[0].set_xlabel('epoch')
    axs[0].legend(loc='lower right')
    # summarize history for loss
    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_title('model loss')
    axs[1].set_ylabel('loss')
    axs[1].set_xlabel('epoch')
    axs[1].legend(loc='upper right')
    return fig, axs