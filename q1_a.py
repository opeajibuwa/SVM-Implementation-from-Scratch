import cvxpy as cp
import numpy as np
import scipy.io as sio
from helper import plot, objective

train_sep = sio.loadmat('/scratch/sagar/Projects/class/cvx/svm/separable_case/train_separable.mat')
test_sep = sio.loadmat('/scratch/sagar/Projects/class/cvx/svm/separable_case/test_separable.mat')

A = train_sep['A']
B = train_sep['B']

X_test = test_sep['X_test']
labels = test_sep['true_labels']

N = A.shape[1]

u = cp.Variable((N,1))
v = cp.Variable((N,1))

obj = cp.Minimize(cp.square(cp.norm(A @ u - B @ v)))
constraints = [cp.sum(u) == 1, u >= 0, cp.sum(v) == 1, v >= 0]

prob = cp.Problem(obj, constraints)
prob.solve(verbose=False)

u_star = np.matmul(A, u.value)
v_star = np.matmul(B, v.value)

w = u_star - v_star

gamma = (np.linalg.norm(u_star)**2 - np.linalg.norm(v_star)**2)/2
# gamma = np.matmul((u_star+v_star).T/2, w)
gamma = gamma.squeeze()

gamma_u = np.matmul(w.T, u_star).squeeze()
gamma_v = np.matmul(w.T, v_star).squeeze()

A_y = np.matmul(A.T, w)
A_y = A_y > gamma
B_y = np.matmul(B.T, w)
B_y = B_y < gamma

predictions = np.matmul(X_test.T, w)
predictions = predictions > gamma

acc = ((predictions.squeeze()*2 - 1) == labels.squeeze())
print("Classification error: {}".format(1- acc.sum()/len(acc)))
print('objective: {}'.format(objective(A, B, u.value, v.value)))
# Plot results
plot(A, B, X_test, labels, gamma, gamma_u, gamma_v, w, save_path='data/q1_a.pdf', title='Q1(a) (separable case)')

