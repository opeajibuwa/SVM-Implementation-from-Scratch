import matplotlib.pyplot as plt
import numpy as np

FONTSIZE=14

def plot(A, B, X_test, labels, gamma, gamma_u, gamma_v, w, save_path=None, title=None):

    x = np.arange(-10, 10)
    y = (gamma - w[0]*x)/w[1]

    xu = np.arange(-10, 10)
    yu = (gamma_u - w[0]*x)/w[1]

    xv = np.arange(-10, 10)
    yv = (gamma_v - w[0]*x)/w[1]


    fig, ax = plt.subplots(1,2, figsize=(10, 5))

    colors = np.concatenate((np.ones(A.shape[1]), np.zeros(B.shape[1])))
    ax[0].scatter(A[0,:], A[1,:], c='green')
    ax[0].scatter(B[0,:], B[1,:], c='blue')


    ax[0].plot(x, y, label='separating hyperplane')
    ax[0].plot(xu, yu, label='supporting hyperplane', linestyle='--', color='red')
    ax[0].plot(xv, yv, linestyle='--', color='red')
    ax[0].set_title('Training examples', fontsize=FONTSIZE)
    ax[0].legend(['separating hyperplane', 'supporting hyperplane'], fontsize=FONTSIZE)

    ax[1].scatter(X_test[0,:100], X_test[1,:100], c='green', label='set A')
    ax[1].scatter(X_test[0,100:], X_test[1,100:], c='blue', label='set B')

    ax[1].legend(fontsize=FONTSIZE)
    ax[1].plot(x, y)
    ax[1].plot(xu, yu, linestyle='--', color='red')
    ax[1].plot(xv, yv, linestyle='--', color='red')
    ax[1].set_title('Testing examples', fontsize=FONTSIZE)

    if title is not None:
        fig.suptitle(title, fontsize=FONTSIZE)

    if save_path is not None:
        fig.savefig(save_path)


def plot_objective_time(obj_list, time_list, title=None, save_path=None):
    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    ax[0].plot(obj_list)
    ax[0].set_xlabel('iteration', fontsize=FONTSIZE)
    ax[0].set_ylabel('objective', fontsize=FONTSIZE)

    ax[1].plot(time_list, obj_list)
    ax[1].set_xlabel('time',fontsize=FONTSIZE)
    ax[1].set_ylabel('objective',fontsize=FONTSIZE)

    if title is not None:
        fig.suptitle(title, fontsize=FONTSIZE)

    if save_path is not None:
        fig.savefig(save_path)
    

def objective(A, B, u, v):
    return np.linalg.norm(np.matmul(A,u) - np.matmul(B, v))**2