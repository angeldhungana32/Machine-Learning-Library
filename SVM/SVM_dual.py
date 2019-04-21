import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from scipy.optimize import Bounds


class SVM_dual_class:
    def __init__(self, x, y, c, r):
        self.X = x
        self.Y = y
        self.C = c
        self.rate = r
        self.support_vectors = None

    def cons1(self, alphas):
        return np.sum(alphas @ self.Y) - 0

    def svm_dual_linear(self):
        alphas = np.ones(len(self.X))
        # Add 1 dimension for bias
        cons = {'type': 'eq', 'fun': self.cons1}
        bnds = Bounds([0.0 for _ in range(len(self.X))],
                      [self.C for _ in range(len(self.X))])
        opt = optimize.minimize(
            self.cost_function,
            alphas,
            args=(self.X, self.Y, self.C),
            method='SLSQP',
            constraints=cons,
            bounds=bnds)
        alpha = opt.x
        w = self.get_weights(alpha, self.X, self.Y)
        b = self.get_bias(w, self.X, self.Y)
        return w, b

    def svm_dual_kernel(self):
        alphas = np.zeros(len(self.X))
        # Add 1 dimension for bias
        cons = {'type': 'eq', 'fun': self.cons1}
        bnds = Bounds([0.0 for _ in range(len(self.X))],
                      [self.C for _ in range(len(self.X))])
        opt = optimize.minimize(
            self.cost_function2,
            alphas,
            args=(self.X, self.Y, self.C),
            method='SLSQP',
            constraints=cons,
            bounds=bnds)
        alpha = opt.x
        self.set_support_vectors(alpha)
        w = self.get_weights(alpha, self.X, self.Y)
        b = self.get_bias(w, self.X, self.Y)
        return w, b

    def set_support_vectors(self, alpha):
        alpha_idx = np.where(alpha > 0)[0]
        support_vectors_ = self.X[alpha_idx, :]
        self.support_vectors = support_vectors_

    def get_weights(self, alpha, X, Y):
        return X.T @ (alpha * Y)

    def get_bias(self, w, X, Y):
        b_tmp = Y - np.dot(w.T, X.T)
        return np.mean(b_tmp)

    def cost_function(self, alphas, X, Y, c):
        jj = -0.5 * np.sum(Y * alphas * np.dot(X, X.T)) - np.sum(alphas)
        return jj

    def cost_function2(self, alphas, X, Y, c):
        y = np.diag(Y)
        sum_alphas = alphas * alphas.T
        sum_xs = self.Kernel(X)
        jj = -0.5 * np.sum(((y.T * y) * sum_xs * sum_alphas)) - np.sum(alphas)
        return jj

    def Kernel(self, X):
        return np.exp((-1.0 * np.linalg.norm(X * X) / self.rate))

    def predict(self, xdata, ydata, theta, b, r):
        yhats = []
        values = [-1, 1]
        for ii in range(len(xdata)):
            xii = xdata[ii]
            yhatii = np.sign(xii.dot(theta) + b)
            yhatii = xii.dot(theta) + b
            yhatii = values[-1] if yhatii >= 0 else values[0]
            yhats.append(yhatii)
        yhats = np.array(yhats)
        counter = 0
        for i in range(len(ydata)):
            if ydata[i] != yhats[i]:
                counter += 1
        return (counter / len(ydata)) * r
