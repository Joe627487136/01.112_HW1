import numpy as np
from numpy.linalg import inv
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

if __name__ == "__main__":
    ifile = open("linear.csv", "rb")
    data = np.genfromtxt(ifile, delimiter=",")
    X1 = data[:, 1]
    X2 = data[:, 2]
    X3 = data[:, 3]
    X4 = data[:, 4]
    Y = data[:, 0]
    n = len(X1)
    Xtest_set = [1.47,1.50,1.52,1.55,1.57,1.60,1.63,1.65,1.68,1.70,1.73,1.75,1.78,1.80,1.83]
    Ytest_set = [52.21,53.12,54.48,55.84,57.20,58.57,59.93,61.29,63.11,64.47,66.28,68.10,69.92,72.19,74.46]

    Sum_y = sum(Y)    # Sy
    Sum_x1y = Compute_double_sum(X1, Y)
    Sum_x2y = Compute_double_sum(X2, Y)
    Sum_x3y = Compute_double_sum(X3, Y)
    Sum_x1 = sum(X1)  # Sx1
    Sum_x2 = sum(X2)  # Sx2
    Sum_x3 = sum(X3)  # Sx3
    Sum_x1sq = Compute_double_sum(X1, X1)
    Sum_x2sq = Compute_double_sum(X2, X2)
    Sum_x3sq = Compute_double_sum(X3, X3)
    Sum_x1x2 = Compute_double_sum(X1, X2)
    Sum_x1x3 = Compute_double_sum(X1, X3)
    Sum_x2x3 = Compute_double_sum(X2, X3)
    a = np.array([[n,Sum_x1,Sum_x2,Sum_x3],[Sum_x1,Sum_x1sq,Sum_x1x2,Sum_x1x3],[Sum_x2,Sum_x1x2,Sum_x2sq,Sum_x2x3],[Sum_x3,Sum_x1x3,Sum_x2x3,Sum_x3sq]])
    matrix1 = inv(a)
    matrix2 = np.array([[Sum_y],[Sum_x1y],[Sum_x2y],[Sum_x3y]])
    ans = np.dot(matrix1,matrix2)
    print(ans)
