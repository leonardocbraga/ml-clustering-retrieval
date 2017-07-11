import numpy as np
import matplotlib.mlab as mlab
import colorsys
from matplotlib import pyplot as plt

def plot_contours(data, means, covs, title):
    plt.figure()
    plt.plot([x[0] for x in data], [y[1] for y in data],'ko') # data

    delta = 0.025
    k = len(means)
    x = np.arange(-2.0, 7.0, delta)
    y = np.arange(-2.0, 7.0, delta)
    X, Y = np.meshgrid(x, y)
    col = ['green', 'red', 'indigo']
    for i in range(k):
        mean = means[i]
        cov = covs[i]
        sigmax = np.sqrt(cov[0][0])
        sigmay = np.sqrt(cov[1][1])
        sigmaxy = cov[0][1]/(sigmax*sigmay)
        Z = mlab.bivariate_normal(X, Y, sigmax, sigmay, mean[0], mean[1], sigmaxy)
        plt.contour(X, Y, Z, colors = col[i])
        plt.title(title)
    plt.rcParams.update({'font.size':16})
    plt.tight_layout()
    plt.show()

def plot_responsibilities_in_RB(img, resp, title):
    N, K = resp.shape
    
    HSV_tuples = [(x*1.0/K, 0.5, 0.9) for x in range(K)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    
    R = img['red']
    B = img['blue']
    resp_by_img_int = [[resp[n][k] for k in range(K)] for n in range(N)]
    cols = [tuple(np.dot(resp_by_img_int[n], np.array(RGB_tuples))) for n in range(N)]

    plt.figure()
    for n in range(len(R)):
        plt.plot(R[n], B[n], 'o', c=cols[n])
    plt.title(title)
    plt.xlabel('R value')
    plt.ylabel('B value')
    plt.rcParams.update({'font.size':16})
    plt.tight_layout()
    plt.show()

def visualize_data(data):
    plt.figure()
    d = np.vstack(data)
    plt.plot(d[:,0], d[:,1],'ko')
    plt.rcParams.update({'font.size':16})
    plt.tight_layout()
    plt.show()

def visualize_loglikelihoods(loglikelihoods):
    plt.figure()
    plt.plot(range(len(loglikelihoods)), loglikelihoods, linewidth=4)
    plt.xlabel('Iteration')
    plt.ylabel('Log-likelihood')
    plt.rcParams.update({'font.size':16})
    plt.tight_layout()
    plt.show()

def plot_top_images(imgs):
    for i in range(5):
        plt.figure()
        plt.imshow(imgs[i].pixel_data)

    plt.show()
