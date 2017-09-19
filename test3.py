import numpy as np
from numpy.linalg import inv
import statsmodels.api as sm
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import csv
import math


def Compute_sqr_sum(x):
    rs=0
    for i in x:
        rs=rs+i*i
    return rs


def Compute_double_sum(Xset, Yset):
    rs=0
    i=0
    for index in range (0,len(Xset)):
        rs = rs+(Xset[index]*Yset[index])
    return rs


def reg_m(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()
    return results


if __name__ == "__main__":
    ifile = open("linear.csv", "rb")
    data = np.genfromtxt(ifile, delimiter=",")
    X1 = data[:, 1]
    X2 = data[:, 2]
    X3 = data[:, 3]
    X4 = data[:, 4]
    Y = data[:, 0]
    n = len(X1)
    x_matrix = [X3,X2,X1]
    print (reg_m(Y, x_matrix).summary())
