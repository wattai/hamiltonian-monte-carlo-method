# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 22:21:46 2018

@author: wattai
"""

# test for Hamiltonian Monte Carlo method.

import numpy as np
import matplotlib.pyplot as plt


def leap_frog(theta, p, h, eps, L):
    for l in range(L):
        p_half = p - eps/2. * h.dpotent(theta)
        theta_next = theta + eps * p_half
        p_next = p_half - eps/2. * h.dpotent(theta_next)
        theta, p = theta_next.copy(), p_next.copy()
    return theta_next, p_next


def Hamiltonian(h, theta, p):
    return h.potent(theta) + np.sum(p**2, axis=0)/2.


class gamma:
    def __init__(self, alpha, lam):
        self.alpha = alpha
        self.lam = lam

    def potent(self, theta):
        return self.lam*theta - (self.alpha - 1.)*np.log(theta)

    def dpotent(self, theta):
        return self.lam - (self.alpha - 1.) / theta


class normal:
    def __init__(self, x):
        self.x = x.copy()
        self.N = self.x.shape[0]

    def potent(self, theta):
        mu = theta[0].copy()  # np.mean(x, axis=0)
        sigma2 = theta[1].copy()  # np.var(x, axis=0)

        out = self.N/2*np.log(sigma2) + 1/(2*sigma2)*np.sum(
                (self.x - mu)**2, axis=0)
        print('potent:', out)
        return out

    def dpotent(self, theta):
        mu = theta[0].copy()  # np.mean(x, axis=0)#
        sigma2 = theta[1].copy()  # np.var(x, axis=0)#

        dmu = -1/sigma2*np.sum(self.x - mu, axis=0)
        dsigma2 = self.N/(2*sigma2) - 1/(2*sigma2**2)*np.sum(
                (self.x - mu)**2, axis=0)
        dtheta = np.array([dmu, dsigma2])
        # print('dpotent:', dtheta)
        return dtheta


class HMC:
    def __init__(self, h, eps, L, T):
        self.n_dim = None

        self.theta = None
        self.p = None
        self.thetas = []
        self.ps = []

        self.h = h
        self.eps = eps
        self.L = L
        self.T = T

    def fit(self, theta):
        self.n_dim = theta.shape[0]
        self.theta = theta
        self.thetas.append(self.theta.copy())
        self.p = np.zeros([self.n_dim])
        self.ps.append(self.p.copy())

        for t in range(T):
            self.p = np.random.randn(n_dim)
            theta_a, p_a = leap_frog(theta=self.theta, p=self.p,
                                     h=self.h, eps=self.eps, L=self.L)
            print('theta_a:', theta_a)
            print('p_a:', p_a)
            H_t = Hamiltonian(h=self.h, theta=self.theta, p=self.p)
            print('H_t:', H_t)
            H_a = Hamiltonian(h=self.h, theta=theta_a, p=p_a)
            print('H_a:', H_a)
            r = np.exp(H_t - H_a)
            print('r:', r)
            update_index = (np.random.uniform(0, 1) <= np.minimum(1, r))
            self.theta[update_index] = theta_a[update_index].copy()

            self.thetas.append(self.theta.copy())
            self.ps.append(self.p.copy())
            print('p:', self.p, 'theta:', self.theta)

    def sample(self):
        return np.array(self.thetas)

    def momentum(self):
        return np.array(self.ps)


if __name__ == '__main__':

    # このデータの平均値と分散をHMC法で推定する.
    x = np.random.normal(170, 7, size=(5000))

    n_dim = 2
    eps = 0.05
    L = 100  # n_iter of Leap-Frog method.
    T = 5000  # n_iter of sample.
    T_burnin = int(0.5*T)  # burn-in period.

    # theta = np.random.uniform(low=0, high=1, size=(n_sample, n_dim))
    theta = 100. * np.ones([n_dim])
    p = np.zeros([n_dim])  # momentum

    # h = gamma(alpha=11, lam=13)
    h = normal(x=x)

    # theta_next, p_next = leap_frog(theta, p, h, eps, L) # Leap-Frog test.
    hmc = HMC(h, eps, L, T)
    hmc.fit(theta)
    thetas = hmc.sample()
    ps = hmc.momentum()

    # plot graphs.
    plt.figure()
    plt.subplot(211)
    plt.plot(thetas)
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$\theta$')
    plt.grid()

    plt.subplot(212)
    theta_map = []
    for i in range(n_dim):
        y_hist, x_hist = np.histogram(thetas[T_burnin:, i],
                                      int(0.5*T),
                                      normed=False)
        theta_map.append(x_hist[np.argmax(y_hist)])
        plt.bar(x=x_hist[:-1], height=y_hist)
    plt.xlabel(r'$\theta$')
    plt.ylabel('hist.')
    plt.grid()

    plt.tight_layout()
    plt.show()

    # print estimated MAP value.
    theta_map = np.array(theta_map)
    print('theta_map:', theta_map)
