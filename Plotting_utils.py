import numpy as np
from colorsys import hsv_to_rgb
import matplotlib.pyplot as plt



def Colorize(z):
    n, m = z.shape
    c = np.zeros((n, m, 3))
    c[np.isinf(z)] = (1, 1, 1)
    c[np.isnan(z)] = (0, 0, 0)
    c[np.abs(z) == 0] = (0, 0, 0)
    idx = ~(np.isinf(z) + np.isnan(z) + (np.abs(z) == 0))
    B = np.abs(z[idx]) / np.max(np.abs(z[idx]))
    A = (np.angle(z[idx]) + np.pi) / (2 * np.pi)
    c[idx] = [hsv_to_rgb(a, b, b) for a, b in zip(A, B)]
    return c

def show_3d(X):
    # abs__,angle__ = np.abs(expanded),np.angle(expanded)

    plt.figure()
    plt.imshow(Colorize(X[X.shape[0]//2,:,:].detach().cpu().numpy()),origin = 'lower')
    plt.title("(x) Dim 0 (z)")
    plt.ylabel(' Dim 1 (x)')
    plt.xlabel(' Dim 2 (y)')

    plt.figure()
    plt.imshow(Colorize(X[:,X.shape[1]//2,:].detach().cpu().numpy()),origin = 'lower')
    plt.title("⨀ Dim 1 (x)")
    plt.ylabel(' Dim 0 (z)')
    plt.xlabel(' Dim 2 (y)')

    plt.figure()
    plt.imshow(Colorize(X[:,:,X.shape[2]//2].detach().cpu().numpy()),origin = 'lower')
    plt.title("(x) Dim 2 (y)")
    plt.ylabel(' Dim 0 (z)')
    plt.xlabel(' Dim 1 (x)')


def show_3d_abs(X):
    abs__ = np.abs(X.detach().cpu().numpy())

    plt.figure()
    plt.imshow(abs__[abs__.shape[0]//2,:,:],origin = 'lower')
    plt.title("(x) Dim 0 (z)")
    plt.ylabel(' Dim 1 (x)')
    plt.xlabel(' Dim 2 (y)')

    plt.figure()
    plt.imshow(abs__[:,abs__.shape[0]//2,:],origin = 'lower')
    plt.title("⨀ Dim 1 (x)")
    plt.ylabel(' Dim 0 (z)')
    plt.xlabel(' Dim 2 (y)')

    plt.figure()
    plt.imshow(abs__[:,:,abs__.shape[0]//2],origin = 'lower')
    plt.title("(x) Dim 2 (y)")
    plt.ylabel(' Dim 0 (z)')
    plt.xlabel(' Dim 1 (x)')
    
def show_3d_ang(X):
    angle__ =np.angle(X.detach().cpu().numpy())

    plt.figure()
    plt.imshow(angle__[angle__.shape[0]//2,:,:],origin = 'lower')
    plt.title("(x) Dim 0 (z)")
    plt.ylabel(' Dim 1 (x)')
    plt.xlabel(' Dim 2 (y)')


    plt.figure()
    plt.imshow(angle__[:,angle__.shape[0]//2,:],origin = 'lower')
    plt.title("⨀ Dim 1 (x)")
    plt.ylabel(' Dim 0 (z)')
    plt.xlabel(' Dim 2 (y)')

    plt.figure()
    plt.imshow(angle__[:,:,angle__.shape[0]//2],origin = 'lower')
    plt.title("(x) Dim 2 (y)")
    plt.ylabel(' Dim 0 (z)')
    plt.xlabel(' Dim 1 (x)')