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

def pg(A, B, X_test, labels_test, show_plot=False, save_path=None, separable=True, nesterov=False, plot_title=None):
    N = A.shape[1]

    u = np.zeros((N,1))
    v = np.zeros((N,1))
    u_old = u.copy()
    v_old = v.copy()

    MAXITER = 200
    STEPSIZE_U = 1/(np.real(np.linalg.eig(np.matmul(A.T, A))[0].max()))
    STEPSIZE_V = 1/(np.real(np.linalg.eig(np.matmul(B.T, B))[0].max()))
    print(STEPSIZE_U, STEPSIZE_V)

    i = 0
    change_iterate = 300
    obj_list = []
    time_list = []
    t1 = time.time()
    
    if separable:
        d = 1
    else:
        d = 0.02

    a_old = 0

    if nesterov:
        EPSILON = -1
    else:
        EPSILON = 1e-5

    while i < MAXITER and change_iterate > EPSILON:
        i = i+1

        a_new = 0.5*(1+np.sqrt(4*a_old**2 +1))    
        a_old = a_new.copy()
        # print(a_new)
        if nesterov:
            t = (a_old - 1)/a_new
        else:
            t = 0

        yu = (1 + t)*u - t*u_old
        yv = (1 + t)*v - t*v_old
    
        u_old = u.copy()
        v_old = v.copy()
        

        yu_grad = 2*np.matmul(np.matmul(A.T, A), yu) - 2*np.matmul(A.T, np.matmul(B,yv))
        yv_grad = 2*np.matmul(np.matmul(B.T, B), yv) - 2*np.matmul(B.T, np.matmul(A,yu))
        
        # gradient step
        u_bar = yu - STEPSIZE_U*yu_grad
        v_bar = yv - STEPSIZE_V*yv_grad

        # u_grad = 2*np.matmul(np.matmul(A.T, A), u) - 2*np.matmul(A.T, np.matmul(B,v))
        # v_grad = 2*np.matmul(np.matmul(B.T, B), v) - 2*np.matmul(B.T, np.matmul(A,u))
        
        # gradient step
        # u_bar = u - STEPSIZE_U*yu_grad
        # v_bar = v - STEPSIZE_V*yv_grad
        
        # projection step
        u_proj = cp.Variable((N,1))
        v_proj = cp.Variable((N,1))
        
        obj_u = cp.Minimize(cp.square(cp.norm(u_bar - u_proj)))
        constraints_u = [cp.sum(u_proj) == 1, u_proj >= 0, u_proj <= d]
        prob_u = cp.Problem(obj_u, constraints_u)
        prob_u.solve(verbose=False)
        u = u_proj.value
        
        obj_v = cp.Minimize(cp.square(cp.norm(v_bar - v_proj)))
        constraints_v = [cp.sum(v_proj) == 1, v_proj >= 0, u_proj <= d]
        prob_v = cp.Problem(obj_v, constraints_v)
        prob_v.solve(verbose=False)
        v = v_proj.value
        
        change_iterate = np.linalg.norm(u-u_old)**2 + np.linalg.norm(v-v_old)**2
        print(i, change_iterate)
        
        obj_list.append(objective(A, B, u, v))
        time_list.append(time.time()-t1)
        # print('iter: {}, obj: {}'.format(i, change_iterate))
        
    u_star = np.matmul(A, u)
    v_star = np.matmul(B, v)

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
    obj_pg_chull, time_pg_chull = pg(A, B, X_test, labels_test, 
                                        show_plot=True,
                                        save_path='data/q2_pg_chull.pdf', 
                                        plot_title='Projected gradient separable case')

    obj_pg_rchull, time_pg_rchull = pg(A_overlap, B_overlap, X_test_overlap, labels_test_overlap, 
                                        show_plot=True,
                                        save_path='data/q2_pg_rchull.pdf', 
                                        plot_title='Projected gradient overlapped case')

    obj_pg_chull_nest, time_pg_chull_nest = pg(A, B, X_test, labels_test, 
                                        show_plot=True,
                                        save_path='data/q2_pg_chull_nest.pdf', 
                                        plot_title='Projected gradient separable case with nesterov accel.', 
                                        nesterov=True)

    obj_pg_rchull_nest, time_pg_rchull_nest = pg(A_overlap, B_overlap, X_test_overlap, labels_test_overlap, 
                                        show_plot=True,
                                        save_path='data/q2_pg_rchull_nest.pdf', 
                                        plot_title='Projected gradient overlapped case with nesterov accel.', 
                                        nesterov=True)




    
    