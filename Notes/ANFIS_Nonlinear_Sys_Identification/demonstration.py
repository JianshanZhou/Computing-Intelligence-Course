# -*- coding: utf-8 -*-
"""
Copyright (C) Thu Dec 01 11:07:31 2016  Jianshan Zhou
Contact: zhoujianshan@buaa.edu.cn	jianshanzhou@foxmail.com
Website: <https://github.com/JianshanZhou>

This program is free software: you can redistribute
 it and/or modify it under the terms of
 the GNU General Public License as published
 by the Free Software Foundation,
 either version 3 of the License,
 or (at your option) any later version.
 
This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY;
 without even the implied warranty of MERCHANTABILITY
 or FITNESS FOR A PARTICULAR PURPOSE.
 See the GNU General Public License for more details.
 You should have received a copy of the GNU General Public License
 along with this program.
 If not, see <http://www.gnu.org/licenses/>.
 
This module carries out some experiments where the ANFIS will be applied
to identify an unknown system given a time series of the system outputs,
and to achieve adaptive nonlinear noise cancellation.
"""
import matplotlib.pyplot as plt
from prepare_data import load_test_data1
from anfis import ANFIS
import numpy as np
import copy
from anfis_mfs import gbellmf
    
def experiment1(training_data,validation_data,all_data):    
    #training_data,validation_data,all_data = load_test_data1()
    # network initialization
    net = ANFIS([4,1])
    net.initialization(training_data)
    init_net = copy.deepcopy(net)
    # train the net
    evaluation_cost, evaluation_accuracy, \
    training_cost, training_accuracy =net.stochastic_gradient_descent(training_data,
                                    validation_data,
                                    eta = 0.1,
                                    mini_batch_size=50,
                                    epoch = 25,
                                    adapt_eta_mode = True,
                                    evaluation_track_flag=True,
                                    training_track_flag=True)

    return evaluation_cost, evaluation_accuracy, \
    training_cost, training_accuracy, net, init_net

def visulization1(evaluation_cost, evaluation_accuracy, \
    training_cost, training_accuracy, init_net, net,\
    training_data,validation_data):
    lc = ['r', 'g', 'b', 'y']
    ls = ['-', '--', ':', '-.']    
    plt.rc('text', usetex=True)
    plt.rc("font",family="serif")
    
    # Show the initial mfs
    mini_batch_size = len(training_data)
    (x,y) = training_data[0]
    input_num = len(x)
    X = x.reshape(1,-1)
    Y = y.reshape(1,-1)
    for k in xrange(mini_batch_size):
        if k>0:
            (x,y) = training_data[k]
            X = np.vstack((X,x.reshape(1,-1)))
            Y = np.vstack((Y,y.reshape(1,-1)))
            
    labelfont = {"family":"serif","size":25}
    plt.figure(3,figsize=(10,8))
    plt.suptitle(r"Initial membership functions of $\displaystyle\mathbf{X}$",\
    fontsize=labelfont["size"])
    for i in range(input_num):
        # for the i-th input
        vmin = np.min(X[::,i])
        vmin = vmin*0.99
        vmax = np.max(X[::,i])
        vmax = vmax*1.01
        input_xi = np.linspace(vmin,vmax,500)
        # the i-th input
        activation1_list = []
        for ji in range(init_net.mf_num_vector[i]):
            params = init_net.premiseParam[i][ji]
            activation1_list.append(np.array([gbellmf(xi,params) for xi in input_xi]))
        plt.subplot(4,1,i+1)
        plt.grid(True)
        for ji in range(init_net.mf_num_vector[i]):
            plt.plot(input_xi,activation1_list[ji],lc[ji]+ls[ji],lw=6.)
        plt.xlabel(r"Input: x{0}".format(i+1))
        plt.ylabel(r"Outputs of MFs")
        plt.legend(loc=0)
    plt.subplots_adjust(hspace=0.39)
        
    plt.figure(4,figsize=(10,8))
    plt.grid(True)
    plt.suptitle(r"Final membership functions of $\displaystyle\mathbf{X}$",\
    fontsize=labelfont["size"])
    for i in range(input_num):
        # for the i-th input
        vmin = np.min(X[::,i])
        vmin = vmin*0.99
        vmax = np.max(X[::,i])
        vmax = vmax*1.01
        input_xi = np.linspace(vmin,vmax,500)
        # the i-th input
        activation1_list = []
        for ji in range(net.mf_num_vector[i]):
            params = net.premiseParam[i][ji]
            activation1_list.append(np.array([gbellmf(xi,params) for xi in input_xi]))
        plt.subplot(4,1,i+1)
        plt.grid(True)
        for ji in range(net.mf_num_vector[i]):
            plt.plot(input_xi,activation1_list[ji],lc[ji]+ls[ji],lw=6.)
        plt.xlabel(r"Input: x{0}".format(i+1))
        plt.ylabel(r"Outputs of MFs")
        plt.legend(loc=0)
    plt.subplots_adjust(hspace=0.39)
        
    # Show the training cost and the validation cost
    labelfont = {"family":"serif","size":25}
    plt.figure(0,figsize=(10,8))
    
    plt.plot(training_cost,'-ob',lw=8.0, ms=16, label="Training cost")
    plt.plot(evaluation_cost,'--or',lw=8.0, ms=16, label="Validation cost")
    plt.grid(True)
    plt.subplots_adjust(top=0.85)
    plt.subplots_adjust(left=0.18)
    plt.legend(prop={"size":labelfont["size"],"family":"serif"},loc=0)  
    xlabelstr = r"Time epoch $t$"
    plt.title(r"ANFIS cost: $\displaystyle\frac{1}{P}\sum_{p=1}^{P}{\Vert"
    r"\mathbf{a}_{p}^{5}-\mathbf{y}_{p}\Vert_2^2}$",fontdict=labelfont)
    plt.xlabel(xlabelstr,fontdict=labelfont)
    plt.ylabel(r"Total cost",fontdict=labelfont)
    plt.xticks(fontsize=labelfont["size"],\
    fontname=labelfont["family"])
    plt.yticks(fontsize=labelfont["size"],
               family=labelfont["family"])
    plt.show()
    
    # Show the training cost and the validation cost
    labelfont = {"family":"serif","size":25}
    plt.figure(1,figsize=(10,8))
    plt.plot(np.asarray(training_accuracy)*100,'-ob',lw=8.0, ms=16, label="Training accuracy")
    plt.plot(np.asarray(evaluation_accuracy)*100,'--or',lw=8.0, ms=16, label="Validation accuracy")
    plt.grid(True)
    plt.subplots_adjust(top=0.85)
    plt.subplots_adjust(left=0.15)
    plt.legend(prop={"size":labelfont["size"],"family":"serif"},loc=0)  
    xlabelstr = r"Time epoch $t$"
    plt.title(r"ANFIS accuracy: "
    r"$\displaystyle\frac{1}{P}"
    r"\sum_{p=1}^{P}{\sum_{l=1}^{m}{\left\vert\frac{a_{l,p}^{5}"
    r"-y_{l,p}}{y_{l,p}}\right\vert}}$",fontdict=labelfont)
    plt.xlabel(xlabelstr,fontdict=labelfont)
    plt.ylabel(r"Relative error [%]",fontdict=labelfont)
    plt.xticks(fontsize=labelfont["size"],\
    fontname=labelfont["family"])
    plt.yticks(fontsize=labelfont["size"],
               family=labelfont["family"])
    plt.show()
    
    # validation
    predictions = np.array([net.inference(x.reshape(-1,1))[0,0] for \
    (x,y) in validation_data])
    actual_outputs = np.array([y[0] for (x,y) in validation_data])
    labelfont = {"family":"serif","size":25}
    plt.figure(2,figsize=(10,8))
    plt.subplots_adjust(left=0.1)
    plt.plot(actual_outputs,'-b',lw=8.0, ms=16, label="Actual")
    plt.plot(predictions,'--r',lw=8.0, ms=16, label="ANFIS")
    plt.grid(True)
    plt.legend(prop={"size":labelfont["size"],\
    "family":"serif"},ncol=2,loc=0)  
    xlabelstr = r"Time epoch $t$"
    ylabelstr = r"System output"
    plt.title("Model validation",fontdict=labelfont)
    plt.xlabel(xlabelstr,fontdict=labelfont)
    plt.ylabel(ylabelstr,fontdict=labelfont)
    plt.xticks(fontsize=labelfont["size"],\
    fontname=labelfont["family"])
    plt.yticks(fontsize=labelfont["size"],
               family=labelfont["family"])
    plt.show()

def carry_out_experiment1():
    training_data,validation_data,all_data = load_test_data1()
    evaluation_cost, evaluation_accuracy, \
        training_cost, training_accuracy, \
        net, init_net=experiment1(training_data,validation_data,all_data)
    visulization1(evaluation_cost, evaluation_accuracy, \
        training_cost, training_accuracy, init_net, net,\
        training_data,validation_data)

      
if __name__ == "__main__":
    carry_out_experiment1()







