# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 09:24:46 2016

@author: zhoujianshan

This module provides some membership functions as well as their corresponding
partial derivatives of every parameters.
"""

import numpy as np
import copy 

def gbellmf(x,params):
    """Generalized bell-shaped membership function:
    GBELLMF(X, [A, B, C]) = 1./((1+ABS((X-C)/A))^(2*B));
    
    Parameters
    ----------
    x: a real number
    params: is a list-like or 1-D array-like vector containing
    parameters of this function.
    
    Returns
    ----------
    output: a real number, also
    """
    if len(params) != 3:
        raise ValueError("Something is wrong with the length of the \
        parameter list!")
    a = params[0]
    b = params[1]
    c = params[2]
    if np.abs(a)<=1e-10:
        print a
        raise("Warning: The parameter a of GBELLMF is approximatively zero!!")
    tmp = ((x-c)/a)**2
    if (tmp==0) and (b==0):
        output = 0.5
    elif (tmp==0) and (b<0):
        output = 0
    else:
        tmp = tmp**b
        output = 1.0/(1+tmp)
    return output
        

def dgbellmf(x,params):
    """Numerically calculate the partial derivatives of the mf parameters.
    Note that here the mathematical definition of the derivative is used to
    numerically approximate the partial derivative:
    df(x)/dx = (f(x+EPSILON)-f(x-EPSILON))/(2*EPSILON)
    According to UFLDL Tutorial: http://deeplearning.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
    EPSILON = 1e-4
    
    Parameters
    ----------
    x: a real number
    params: is a list-like or 1-D array-like vector containing
    parameters of this function.
    
    Returns
    ----------
    dParams: a 1-D array, shape (len(params),)
    """
    if len(params) != 3:
        raise ValueError("Something is wrong with the length of the \
        parameter list!")
    EPSILON = 1e-4
    dParams = np.zeros((len(params),))
    for i in range(len(params)):
        params1 = copy.deepcopy(params)#params should not be changed!!!!!!!
        params2 = copy.deepcopy(params)
        params1[i] = params1[i]+EPSILON
        params2[i] = params2[i]-EPSILON
        dParams[i] = (gbellmf(x,params1)-gbellmf(x,params2))/(2*EPSILON)
    return dParams