import pandas as pd
import numpy as np
from pyrsistent import T

class LinUCB(object):
    def __init__(self):
        self.phis = np.array([[arm[0], arm[1], ///, 1] for arm in arms]).T
        self.alpha = 1
        self.sigma = 1
        self.A = np.identity(self.phis.shape[0])
        self.b = np.zeros((self.phis.shape[0], 1))

    def get_arm(self):
        inv_A = np.linalg.inv(self.A)
        mu = inv_A.dot(self.b)
        S = inv_A
        pred_mean = self.phis.T.dot(mu)
        pred_var = self.phis.T.dot(S).dot(self.phis)
        ucb = pred_mean.T + self.alpha * np.sqrt(np.diag(pred_var))
        return np.argmax(ucb)

    def sample(self, arm_index, reward):
        phi = self.phis[:, [arm_index]]
        self.b = self.b + phi * reward / (self.sigma ** 2)
        self.A = self.A + phi.dot(phi.T) / (self.sigma ** 2)