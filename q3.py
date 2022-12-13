import cvxpy as cp
import numpy as np
import scipy.io as sio
from helper import *
import time

train_sep = sio.loadmat('/scratch/sagar/Projects/class/cvx/svm/separable_case/train_separable.mat')
test_sep = sio.loadmat('/scratch/sagar/Projects/class/cvx/svm/separable_case/test_separable.mat')
train_overlap = sio.loadmat('/scratch/sagar/Projects/class/cvx/svm/overlap_case/train_overlap.mat')
test_overlap = sio.loadmat('/scratch/sagar/Projects/class/cvx/svm/overlap_case/test_overlap.mat')

A = train_sep['A']
B = train_sep['B']

X_test = test_sep['X_test']
labels_test = test_sep['true_labels']

A_overlap = train_overlap['A']
B_overlap = train_overlap['B']

X_test_overlap = test_overlap['X_test']
labels_test_overlap = test_overlap['true_labels']


def project_separable(x):
    x[x<0] = 0
    return x

def project_overlap(x, d):
    x[x<0] = 0
    x[x>d] = d
    return x


def admm(A, B, X_test, labels_test, separable=True, plot_title=None, show_plot=True, save_path=None):
    N = A.shape[1]
    MAXITER = 200
    RHO = 1000

    x1 = np.zeros((2*N, 1))
    x2 = np.zeros((2*N, 1))
    y = np.zeros((2*N+2, 1))

    one = np.ones((N, 1))
    zero = np.zeros((N, 1))

    C = np.concatenate((np.eye(2*N), np.concatenate((one.T, zero.T), axis=1), np.concatenate((zero.T, one.T), axis=1)))
    D = np.concatenate((np.eye(2*N), np.zeros((2, 2*N))), axis=0)
    e = np.concatenate((np.zeros((2*N,1)), np.ones((2, 1))), axis=0)

    P = np.concatenate((A, -B), axis=1)

    Q = np.matmul(P.T, P) + (RHO/2)*np.matmul(C.T, C)
    Q_inv = np.linalg.inv(Q)

    obj_list = []
    time_list = []
    t1 = time.time()
    i=0
    d = 0.02
    x2_old = x2.copy()
    primal_residual = np.inf
    dual_residual = np.inf
    while i<MAXITER and (np.linalg.norm(primal_residual)>0.001 or np.linalg.norm(dual_residual)>5):
        i += 1
        
        b = np.matmul((RHO)*C.T, - y + np.matmul(D, x2) + e)
        
        x1 = 0.5*np.matmul(Q_inv, b)
        if separable:
            x2 = project_separable(y[:2*N, :] + x1)
        else:
            x2 = project_overlap(y[:2*N, :] + x1, d)

        primal_residual = np.matmul(C, x1) - np.matmul(D, x2) - e 
        dual_residual = RHO*np.matmul(np.matmul(C.T, -D), x2 - x2_old)

        x2_old = x2.copy()
        y = y + primal_residual

        
        
        obj_list.append(objective(A, B, x1[:100, :], x1[100:, :]))
        time_list.append(time.time()-t1)
        print('iter {}, obj: {}'.format(i, obj_list[-1]), np.linalg.norm(primal_residual), np.linalg.norm(dual_residual))
        
    u_star = np.matmul(A, x2[:100,:])
    v_star = np.matmul(B, x2[100:,:])

    w = u_star - v_star

    gamma = (np.linalg.norm(u_star)**2 - np.linalg.norm(v_star)**2)/2
    # gamma = np.matmul((u_star+v_star).T/2, w)
    gamma = gamma.squeeze()

    gamma_u = np.matmul(w.T, u_star).squeeze()
    gamma_v = np.matmul(w.T, v_star).squeeze()

    predictions = np.matmul(X_test.T, w)
    predictions = predictions > gamma

    acc = ((predictions.squeeze()*2 - 1) == labels_test.squeeze())
    print("Classification error: {}".format(1- acc.sum()/len(acc)))

    if show_plot:
        plot(A, B, X_test, labels_test, gamma, gamma_u, gamma_v, w, save_path=save_path, title=plot_title)
        plot_objective_time(obj_list, time_list, title=plot_title)

    return obj_list, time_list

if __name__=='__main__':
    admm(A, B, X_test, labels_test, separable=True, plot_title=None, show_plot=True, save_path=None)