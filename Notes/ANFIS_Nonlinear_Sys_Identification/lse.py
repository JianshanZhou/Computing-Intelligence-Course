# -*- coding: utf-8 -*-
"""
Copyright (C) Sat Nov 26 17:44:07 2016  Jianshan Zhou
Contact: zhoujianshan@buaa.edu.cn	jianshanzhou@foxmail.com
Website: <https://github.com/JianshanZhou>
 
This module implements a Least Square Estimator (LSE) for solving
argmin:||AX-B||^2 given the data (A,B).
"""

import numpy as np

def LSE(A,B,X):
    """Perform a sequential algorithm to solve the LSE of
    ||AX-B||^2, which can be easily implemented in an online learning fashion.
    
    Parameters
    ----------
    A: a 2-D array, shape (P,M)
    B: a 1-D array, shape (P,)
    
    Returns
    X: a 2-D array, shape (M,1)    
    """
    (P,M) = A.shape
    # initialize Si
    gamma = 1e8
    S = gamma*np.identity(M,dtype=float)
    for i in range(P):
        # the i-th row vector of A        
        a_transpose = A[i,::]
        # keep it as a 2-D array
        a_transpose = a_transpose.reshape((1,len(a_transpose)))
        # the i-th element of B
        b_transpose = B[i]
        den = 1.0+np.dot(a_transpose,np.dot(S,a_transpose.T))
        num = np.dot(np.dot(S,a_transpose.T),np.dot(a_transpose,S))
        S = S - num/den
        X = X + np.dot(S,a_transpose.T)*(b_transpose-np.dot(a_transpose,X))
    return X

def LSE2(a, X, Y, solution):
    """Similar to LSE but receiving different inputs.
    
    Parameters
    ----------
    a: a 1-D array, shape (N,)
    X: a 2-D array, shape (P,n)
    Y: a 1-D array, shape (P,)
    solution: a 2-D array, shape (M,1) where M=N*n
    
    Returns
    solution: a 2-D array, shape (M,1) where M=N*n
    """
    N = len(a)
    (P,n) = X.shape
    M = N*n
    
    # initialize Si
    gamma = 1e8
    S = gamma*np.identity(M,dtype=float)
    for i in range(P):
        # the i-th row vector of A        
        a_transpose = np.hstack((ak*X[i,::] for ak in a))
        # keep it as a 2-D array
        a_transpose = a_transpose.reshape((1,len(a_transpose)))
        # the i-th element of B
        b_transpose = Y[i]
        den = 1.0+np.dot(a_transpose,np.dot(S,a_transpose.T))
        num = np.dot(np.dot(S,a_transpose.T),np.dot(a_transpose,S))
        S = S - num/den
        solution = solution \
        + np.dot(S,a_transpose.T)*(b_transpose-np.dot(a_transpose,solution))
    return solution


def LSE3(A, X, Y, solution):
    """Similar to LSE but receiving different inputs.
    
    Parameters
    ----------
    A: a list of lists, shape (P,N)
    X: a 2-D array, shape (P,n)
    Y: a 1-D array, shape (P,)
    solution: a 2-D array, shape (M,1) where M=N*n
    
    Returns
    solution: a 2-D array, shape (M,1) where M=N*n
    """
    N = len(A[0])
    (P,n) = X.shape
    M = N*n
    
    # initialize Si
    gamma = 1e8
    S = gamma*np.identity(M,dtype=float)
    for i in range(P):
        # the i-th row vector of the coef. matrix
        a = A[i]       
        a_transpose = np.hstack((ak*X[i,::] for ak in a))
        # keep it as a 2-D array
        a_transpose = a_transpose.reshape((1,len(a_transpose)))
        # the i-th element of B
        b_transpose = Y[i]
        den = 1.0+np.dot(a_transpose,np.dot(S,a_transpose.T))
        num = np.dot(np.dot(S,a_transpose.T),np.dot(a_transpose,S))
        S = S - num/den
        solution = solution \
        + np.dot(S,a_transpose.T)*(b_transpose-np.dot(a_transpose,solution))
    return solution


def data_format(a,X,Y):
    """This function is used to rearrange the data representation
    into a compact matrix form.
    
    Parameters
    ----------
    a: a 1-D array, shape (N,)
    X: a 2-D array, shape (P,n)
    Y: a 1-D array, shape (P,)
    
    Returns
    ----------
    A: a 2-D array, shape (P,N*n)
    
    """
    (P,n) = X.shape
    x = X[0,::] # a 1-D array with shape (1,n)
    A = [ak*x for ak in a]
    A = np.asarray(A)
    A = A.reshape(1,-1)
    for i in range(P):
        if i !=0:
            y = np.asarray([ak*X[i,::] for ak in a])
            y = y.reshape(1,-1)
            A = np.vstack((A,y))
    return A
    

def vectorize(W):
    """This function transforms a 2-D array, i.e., a matrix, 
    into a column vector.
    
    Parameters
    ----------
    W: a 2-D array, shape (M,N)
    
    Returns
    ----------
    w: a 2-array, shape (M*N,1)
    """
    return W.reshape(-1,1)

    
if __name__ == "__main__":
    # perform a test on this LSE algorithm
    x = np.array([0, 1, 2, 3])
    y = np.array([-1, 0.2, 0.9, 2.1])
    A = np.vstack([x, np.ones(len(x))]).T
    print A
    m = np.linalg.lstsq(A,y)[0]
    print m
    
    solution = np.zeros((2,1),dtype=float)
    s = LSE(A,y,solution)
    print s
    
    print np.linalg.norm(s-m.reshape(s.shape))
    
    a = np.array([1.0])
    print a
    print A
    B = data_format(a,A,y)
    print B
    
    print "-----------"
    print vectorize(A)
    print "-------------"
    print LSE2(a, A, y, solution)
    
    
        
