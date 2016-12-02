# -*- coding: utf-8 -*-
"""
Copyright (C) Sun Nov 27 01:54:10 2016  Jianshan Zhou

Contact: zhoujianshan@buaa.edu.cn	jianshanzhou@foxmail.com

Website: <https://github.com/JianshanZhou>

 
This module builds the Adaptive Network-based Fuzzy 
Inference System (ANFIS).

References
-----------
Jang, J. S. R. (1993). Anfis: adaptive-network-based fuzzy inference system.
 IEEE Transactions on Systems Man & Cybernetics, 23(3), 665-685.
"""
# The standard modules
import random
import copy

# The third party modules
import numpy as np

# The supplementary modules
import lse as lse
from anfis_mfs import gbellmf, dgbellmf


# The objective function class
class QuadraticC(object):
    @staticmethod
    def C(a,y):
        """The cost function
        
        Parameters
        ----------
        a: the output of the last layer of the ANFIS
        y: the desired output
        a and y should be in the same size, and in 2-D array form
        
        Returns
        ----------
        the quadratic cost, a positive number
        """
        return 0.5*np.linalg.norm(a-y)**2
        
    @staticmethod
    def delta(a,y):
        """Calculate the error rate dC/da for a given training data for each node
        output.
        
        Parameters
        ----------
        a: the output of the last layer of the ANFIS
        y: the desired output
        
        Returns
        ----------
        the error rate in size of a, 2-D array, shape(m,1)
        """
        return (a-y)


# Main ANFIS class
class ANFIS(object):
    
    def __init__(self, in_out_num, 
                 mf_num_vector=None, 
                 cost=QuadraticC):
        """The initial vector of numbers of MF associated with each input.
        In this stage, I consider a general multiple-input-multiple-output adaptive
        network, i.e., a MIMO ANFIS.
        """
        if (len(in_out_num) !=2):
            raise ValueError("Something is wrong with the numbers of inputs\
            and outputs!")
            
        if mf_num_vector:
            self.mf_num_vector = mf_num_vector
        else:
            self.mf_num_vector = [2 for i in range(in_out_num[0])]
            
        self.cost = cost
        self.size = (in_out_num[0],
                     np.sum(self.mf_num_vector),
                     np.prod(self.mf_num_vector),
                     np.prod(self.mf_num_vector),
                     np.prod(self.mf_num_vector)*in_out_num[1],
                     in_out_num[1]) # there are totally 6 layers in ANFIS
        print "A %d-%d-%d-%d-%d-%d ANFIS is initialized!"%self.size

    def initialization(self, training_data):
        """Initialize the premise and the consequent parameters of the ANFIS.
        
        Parameters
        ----------
        training_data: a list of tuples each denoting sample pair (x,y)
        len(traning_data) = p
        x is a 1-D array, shape (n,), and y is also a 1-D array, shape(m,)
        where P is the total sample number, n the input number and m the output number.
        """
        premiseParam, consequentParam = generateFIS(training_data, self.mf_num_vector)
        self.premiseParam = premiseParam
        self.consequentParam = consequentParam
        for i in range(self.size[0]):
            print "The {0}-th input has {1} membership functions each with {2} premise \
            parameters".format(i+1,len(self.premiseParam[i]),len(self.premiseParam[i][0]))
        # Generate the mapped indices
        self.indices_2nd_layer = index_of_2ndLayer(self.mf_num_vector)

    def activations_3rd_layer(self, x):
        """Given a 2-D array and evaluate the outputs of the 3rd layer in the ANFIS.
        
        Parameters
        ----------
        x: a 2-D array, shape(n,1)
        
        Returns
        ----------
        activation1: a list of lists each containing the membership function 
        outputs of an input
        
        activation2: a list, shape (M,)
        
        activation3: a list, length N
        """
        # Ensure 2-D
        if len(x.shape)<2:
            x = x.reshape(-1,1)
        
        # the activations of the 1st layer, 2-D
        activation1 = [[gbellmf(x[i,0],self.premiseParam[i][ji]) for ji \
        in range(self.mf_num_vector[i])] for i in range(self.size[0])]
                
        # the activations of the 2nd layer, 1-D
        activation2 = []
        for ind in self.indices_2nd_layer:
            a2_tmp = np.prod([activation1[i][ind[i]] for i in range(len(ind))])
            activation2.append(a2_tmp)
        
        # the activations of the 3rd layer, 1-D
        sum_a2 = np.sum(np.exp(activation2))
        activation3 = [np.exp(a2_p)/sum_a2 for a2_p in activation2] # a list
            
        return activation1, activation2, activation3

    def forwardpass(self, x):
        """Given a 2-D array and infere the outputs of each layer in by using the ANFIS.
        
        Parameters
        ----------
        x: a 2-D array, shape(n,1)
        
        Returns
        ----------
        activation1: a list of lists each containing the membership function 
        outputs of an input
        
        activation2: a list, shape (M,)
        
        activation3: a list, shape (N,) where N=m1*m2*...*mn, the total rules number
        
        activation4: the activations of the 4th layer, 2-D list, i.e., a list of lists
        each corresponding to an output channel, shape (m, N)
        
        activation5: a 2-D array containing m outputs, shape (m,1)
        """
        activation1, activation2, activation3 = self.activations_3rd_layer(x)
        X = np.vstack((np.array([[1.]]),x)) # X shape (n+1,1)
        # the activations of the 4th layer, 2-D
        activation4 = []
        for l in range(self.size[-1]):
            # for the l-th output
            a4_tmp = []
            for q in range(len(activation3)):
                consequent_params = self.consequentParam[l][q,::]
                # Ensure 2-D
                val = (np.dot(consequent_params.reshape(1,-1),X)) #2-D
                a4_tmp.append(activation3[q]*val[0,0])
            activation4.append(copy.deepcopy(a4_tmp))
        
        # the outputs of the 5th layer
        activation5 = np.array([np.sum(activation4[l]) for l in range(self.size[-1])])
        activation5 = activation5.reshape(-1,1)
        return activation1, activation2, activation3, activation4, activation5        

    def inference(self,x):
        """Given a 2-D array and infere the output by using the ANFIS.
        
        Parameters
        ----------
        x: a 2-D array, shape(n,1)
        
        Returns
        ----------
        y: a 2-D array, shape (m,1)
        """
        activation1, activation2, \
        activation3, activation4, activation5 = self.forwardpass(x)
        return activation5 # output 2-D, shape(m,1)

    def total_cost(self, evaluation_data):
        """Calculate the total cost on the given data set.
        
        Parameters
        ----------
        evaluation_data: a list of tuples each (x,y)
        where x is 1-D array and y is also 1-D array
        
        Returns
        ----------
        Cost, a real number
        """
        Cost = 0.0
        for (x,y) in evaluation_data:
            output = self.inference(x.reshape(-1,1))
            Cost += (self.cost.C(output,y.reshape(-1,1)))
        return Cost/len(evaluation_data)
        
    def average_accuracy(self,evaluation_data):
        """Calculate the averaged cost on the given data set.
        
        Parameters
        ----------
        evaluation_data: a list of tuples each (x,y)
        where x is 1-D array and y is also 1-D array
        
        Returns
        ----------
        avgAcc
        """
        Acc = 0.0
        m = len(evaluation_data[0][1])
        for (x,y) in evaluation_data:
            output = self.inference(x.reshape(-1,1)) #2-D
            acc = 0.0
            for l in range(m):
                ol = output[l,0]
                yl = y[l]
                if (np.abs(yl) == 0) and (np.abs(ol)==0):
                    acc += 1.
                elif (np.abs(yl) == 0) and (np.abs(ol)!=0):
                    acc += 0.0
                else:
                    acc += np.abs((yl-ol)/yl)  
            acc = acc/m
            Acc+=acc            
        return Acc/len(evaluation_data)
        
    def stochastic_gradient_descent(self,training_data0,
                                    validation_data0=None,
                                    eta = 10.0,
                                    mini_batch_size=100,
                                    epoch = 50,
                                    adapt_eta_mode = True,
                                    evaluation_track_flag=False,
                                    training_track_flag=False):
        """Do the SGD algorithm to train the ANFIS.
        
        Parameters
        ----------
        training_data: a list of tuples each denoting a sample pair (x,y)
        x is a 1-D array in shape (n,), and y is also a 1-D array in shape (m,)
        n and m are the numbers of inputs and outputs per sample, respectively.
        validation_data: a list of tuples similar to training_data
        
        eta: the learning rate, a real positive number
        mini_batch_size: the size of batched samples
        epoch: training epoch number
        {evaluation, training}_track_flag: a flag, bool
        
        Returns
        ----------
        evaluation_cost_trace: a 1-D array recording the cost evaluated on evaluation data
        at the end of each epoch, an empty list if its corresponding flag is False
        training_cost_trace: a 1-D array recording the cost on the training data,
        an empty list if its corresponding flag is False
        """
        training_data = copy.deepcopy(training_data0)
        validation_data = copy.deepcopy(validation_data0)
        evaluation_cost = []
        evaluation_accuracy =[]
        training_cost = []
        training_accuracy = []
        total_sample_num = len(training_data)
        track_4points = np.zeros((5,))
        t_count = 0
        for t in xrange(epoch):
            random.shuffle(training_data)
            mini_batch_list = [
                training_data[j:j+mini_batch_size]
                for j in xrange(0,total_sample_num,mini_batch_size)]
            for mini_batch in mini_batch_list:
                self.update_ANFIS(mini_batch,eta,adapt_eta_mode)
#                # adapt the learning rate by using two heuristic rules
#                eval_cost_mini_batch = self.total_cost(mini_batch)
#                if t_count<5:
#                    track_4points[t_count] = eval_cost_mini_batch
#                else:
#                    for ti in range(4):
#                        track_4points[ti] = track_4points[ti+1]
#                    track_4points[-1] = eval_cost_mini_batch
#                if (track_4points[1]>track_4points[2]) \
#                and (track_4points[2]>track_4points[3]) \
#                and (track_4points[3]>track_4points[4]):
#                    eta = eta*(1.1)
#                elif (track_4points[0]<track_4points[1]) \
#                and (track_4points[1]>track_4points[2]) \
#                and (track_4points[2]<track_4points[3]) \
#                and (track_4points[3]>track_4points[4]):
#                    eta = eta*(0.9)
#                t_count += 1
                    
            print "#Complete the epoch: {0}/{1}".format(t,epoch)
            if training_track_flag:
                eval_cost = self.total_cost(training_data)
                eval_acc = self.average_accuracy(training_data)
                print "*Total training cost: {0}".format(eval_cost)
                print "Averaged training accuracy: {0}".format(eval_acc)
                training_cost.append(eval_cost)
                training_accuracy.append(eval_acc)
                # adapt the learning rate by using two heuristic rules
                eval_cost_mini_batch = eval_cost
                if t_count<5:
                    track_4points[t_count] = eval_cost_mini_batch
                else:
                    for ti in range(4):
                        track_4points[ti] = track_4points[ti+1]
                    track_4points[-1] = eval_cost_mini_batch
                if (track_4points[1]>track_4points[2]) \
                and (track_4points[2]>track_4points[3]) \
                and (track_4points[3]>track_4points[4]):
                    eta = eta*(1.1)
                elif (track_4points[0]<track_4points[1]) \
                and (track_4points[1]>track_4points[2]) \
                and (track_4points[2]<track_4points[3]) \
                and (track_4points[3]>track_4points[4]):
                    eta = eta*(0.9)
                t_count += 1
            if validation_data:
                eval_cost = self.total_cost(validation_data)
                eval_acc = self.average_accuracy(validation_data)
                print "*Total validation cost: {0}".format(eval_cost)
                print "Averaged valiation accuracy: {0}".format(eval_acc)
                if evaluation_track_flag:
                    evaluation_cost.append(eval_cost)
                    evaluation_accuracy.append(eval_acc)
            print " "   
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
     
    def update_ANFIS(self,mini_batch, eta, adapt_eta_mode=True):
        """Update ANFIS
        
        Parameters
        ----------
        mini_batch: a list of tuples
        eta: a learning rate
        """
        # rearrange the training data into X and Y
        mini_batch_size = len(mini_batch)
        (x,y) = mini_batch[0]
        X = x.reshape(1,-1)
        Y = y.reshape(1,-1)
        for k in xrange(mini_batch_size):
            if k>0:
                (x,y) = mini_batch[k]
                X = np.vstack((X,x.reshape(1,-1)))
                Y = np.vstack((Y,y.reshape(1,-1)))
        # do main learning algorithm        
        dpremiseParam = self.process_mini_batch(X,Y)
        
        # adapt the learning rate
        if adapt_eta_mode:
            den = 0.0
            for i in range(self.size[0]):
                for ji in range(self.mf_num_vector[i]):
                    den += (np.linalg.norm(dpremiseParam[i][ji]))**2
            den = np.sqrt(den)
            if den<=1e-4:
                lmbda = eta
            else:
                lmbda =eta/den
        else:
            lmbda = eta
        
        # update the premise parameters
        for i in range(self.size[0]):
            for ji in range(self.mf_num_vector[i]):
                self.premiseParam[i][ji] = self.premiseParam[i][ji] \
                -(lmbda/mini_batch_size)*dpremiseParam[i][ji]
        
   
    def process_mini_batch(self,mini_batch_X, mini_batch_Y):
        """Calculate the error rates of all the premise parameters over the
        mini_batch data by using backpropagation.
        
        Parameters
        -----------
        activation1: a list of lists each containing the membership function 
        outputs of an input, shape (n,M), where M = m1+m2+...+mn, the total number
        of the mfs
        
        activation2: a list, shape (M,)
        
        activation3: a list, shape (N,) where N=m1*m2*...*mn, the total rules number
        
        activation4: the activations of the 4th layer, 2-D list, i.e., a list of lists
        each corresponding to an output channel, shape (m, N)
        
        activation5: a 2-D array containing m outputs, shape (m,1)
        
        mini_batch_X: a 2-D array, shape (mini_batch_size,n)
        mini_batch_Y: a 2-D array, shape (mini_batch_size,m)
        
        Returns
        ----------
        dpremiseParameter in the same shape with that of self.premiseParam,
        a list of lists each containing the error rates of mfs
        """
        dpremiseParameter = []
        for i in range(self.size[0]):
            dpremiseParameter.append([np.zeros(self.premiseParam[i][ji].shape) for ji in range(self.mf_num_vector[i])])
        
        activation1_list = []
        activation2_list = []
        activation3_list = []
        activation4_list = []
        activation5_list = []
        for x in mini_batch_X:
            # forward pass
            act1, act2, \
            act3, act4, act5 = self.forwardpass(x.reshape(-1,1)) 
            #act1, act2, act3 = self.activations_3rd_layer(x.reshape(-1,1))
            activation3_list.append(copy.deepcopy(act3))
            activation1_list.append(copy.deepcopy(act1))
            activation2_list.append(copy.deepcopy(act2))
            activation4_list.append(copy.deepcopy(act4))
            activation5_list.append(copy.deepcopy(act5))
        
        # update the consequent parameters with fixed premise parameters
        self.batch_update_consequentParam(activation3_list,
                                          mini_batch_X,mini_batch_Y) 
        index = 0                                  
        # update the premise parameters with fixed consequent parameters
        for x, y in zip(mini_batch_X,mini_batch_Y):
            # reshape x and y into 2-D
            x = x.reshape(-1,1)
            y = y.reshape(-1,1)
            
            # error rates per sample
            dP = self.backpropagation(activation1_list[index], \
                                      activation2_list[index], \
                        activation3_list[index], \
                        activation4_list[index], \
                        activation5_list[index], \
                        x, y)
            index += 1
            #raise("HELLO-0")
            for i in range(self.size[0]):
                for ji in range(self.mf_num_vector[i]):
                    dpremiseParameter[i][ji] = dpremiseParameter[i][ji] + dP[i][ji]
        #show_preParam(self,dpremiseParameter)            
        return dpremiseParameter    
    
    def backpropagation(self, activation1, activation2,
                        activation3, activation4, activation5,
                        x, y):
        """Perform the backpropagation algorithm to derive the error rates of
        each premise parameters in the 1st layer. a single input-output pair is
        given.
        
        Parameters
        ----------
        activation1: a list of lists each containing the membership function 
        outputs of an input, shape (n,M), where M = m1+m2+...+mn, the total number
        of the mfs
        
        activation2: a list, shape (M,)
        
        activation3: a list, shape (N,) where N=m1*m2*...*mn, the total rules number
        
        activation4: the activations of the 4th layer, 2-D list, i.e., a list of lists
        each corresponding to an output channel, shape (m, N)
        
        activation5: a 2-D array containing m outputs, shape (m,1)
        
        x: a 2-D array, shape (n,1)
        y: a 2-D array, shape (m,1)
        
        Returns
        -----------        
        a list in the same shape with that of activation1
        containing the error rates of premise parametrs.
        """
        X = np.vstack((np.array([[1.0]]),x)) #shape (n+1,1)        
        
        #M = np.sum(self.mf_num_vector)
        N = np.prod(self.mf_num_vector)
        
        # the 5-th layer
        delta5 = self.cost.delta(activation5, y) # a 2-D array in shape (m,1)

        # the 4-th layer
        delta4 = []
        for l in range(self.size[-1]):
            tmp_delta4 = []
            for q in range(N):
                tmp_delta4.append(delta5[l,0])
            delta4.append(copy.deepcopy(tmp_delta4)) # like activation4
        
        # the 3rd layer
        delta3 = []
        for q in range(N):
            tmp3 = 0.0
            for l in range(self.size[-1]):
                w = self.consequentParam[l][q,::]
                tmp3 += delta4[l][q]*(np.dot(w.reshape(1,-1),X)[0,0]) # Note that tmp3 is a 2-D array-like variable!!!
            delta3.append(tmp3)
        delta3 = np.asarray(delta3)

        # the 2nd layer
        sum_a2 = np.sum(np.exp(activation2))
        delta2 = []
        for q in range(N):
            tmp_da2 = 0.0
            for s in range(N):
                if s==q:
                    tmp_da2 += (delta3[s])*(np.exp(activation2[q])*((sum_a2-np.exp(activation2[q])))/(sum_a2**2))
                else:
                    tmp_da2 += (delta3[s])*((-np.exp(activation2[s])*np.exp(activation2[q]))/(sum_a2**2))
            delta2.append(tmp_da2)

        # the 1st layer
        delta1 = []
        for i in range(self.size[0]):
            tmp_delta1 = []
            for ji in range(self.mf_num_vector[i]):
                inds = get_indices_of_2ndLayer(i,ji,self.indices_2nd_layer)
                tmp1 = 0.0
                for ind_2nd_layer_node in inds:
                    inds_tuple = self.indices_2nd_layer[ind_2nd_layer_node]
                    
                    partial1_vector = [activation1[i_prime][inds_tuple[i_prime]] \
                    for i_prime in range(self.size[0]) if i_prime != i]
                    
                    tmp1 += delta2[ind_2nd_layer_node]*(np.prod(partial1_vector))
                tmp_delta1.append(tmp1)
            delta1.append(copy.deepcopy(tmp_delta1))
        # the deltas of premise parameters in the same shape with that of premiseParam
        dpremiseParam = []
        for i in range(self.size[0]):
            dpremiseParam.append([delta1[i][ji]*dgbellmf(x[i,0],self.premiseParam[i][ji]) for ji \
                                                 in range(self.mf_num_vector[i])])
        return dpremiseParam

    def batch_update_consequentParam(self, activation3_list, mini_batch_X, mini_batch_Y):
        """Update the consequent parameters by using the least square estimator.
        The update is carried out with batch activations.        
        
        Parameters
        ----------
        activation3: a list of lists each containing the activations of the 3rd layer at each sample
        mini_batch_X: a 2-D array, shape (mini_batch_size,n)
        mini_batch_Y: a 2-D array, shape (mini_batch_size,m)
        n is the inputs number and m the outputs number.
        
        Returns
        ----------
        dpremiseParam: a list of lists each containing the error rates of mfs
        its shape is the same to that of self.premiseParam
        """
        (mini_batch_size,n) = mini_batch_X.shape
        mini_batch_x = np.hstack((np.ones((mini_batch_size,1), dtype=float),mini_batch_X))
        
        for l in range(self.size[-1]):
            mini_batch_y = mini_batch_Y[::,l] # a 1-D array, shape(mini_batch_size,)
            
            # update the consequent parameters related to the l-th output
            consequentParam = self.consequentParam[l] # obtain the parameters
            # vectorize this form
            solution = lse.vectorize(consequentParam)
            # do lse
            solution = lse.LSE3(activation3_list,mini_batch_x,mini_batch_y,solution)
            # update the l-th consequent parameters
            self.consequentParam[l] = solution.reshape(consequentParam.shape)


# Miscellaneous functions
def generateFIS(mini_batch,mf_num_vector):
    """This function creates an initial Sugeno-type FIS
    for ANFIS training by performing
    a grid partiion on the given data. In this function, 
    the generalized bell-shaped function,
    i.e.,f(x;a,b,c) = 1/(1+((x-c)/a)^(2b)), 
    is adopted as the membership function of each input, and
    a linear weighed sum, i.e. sum_{i=1 to n}{wi*xi}+r, is adopted as the 
    membership function of a single output.
    
    Parameters
    ----------
    mini_batch: a list of tuples each denoting sample pair (x,y)
    mf_num_vector: a list of mf numbers
    
    Returns
    ----------
    consequentParam: a list of 2-D arrays each with shape (N,n+1) given N rules and n inputs. This
    consequentParam is arranged as [[w_mi]], m=0,...,N-1; i=0,..,n
    premiseParam: a list of lists each containing the parameter arrays of membership
    functions, like [[1-D-array,...,1-D-array],...]
    
    the parameters set of the i-th node's j-th membership function is indexed by
    premiseParam[i][j], which is an array containing a, b, and c
    consequentParam[m,i] denotes the weight of the m-th rule's i-th input
    """   
    mini_batch_size = len(mini_batch)
    (x,y) = mini_batch[0]
    X = x.reshape(1,-1)
    Y = y.reshape(1,-1)
    for k in xrange(mini_batch_size):
        if k>0:
            (x,y) = mini_batch[k]
            X = np.vstack((X,x.reshape(1,-1)))
            Y = np.vstack((Y,y.reshape(1,-1)))
    return generateFIS2(X, Y, mf_num_vector)


def generateFIS2(X, Y, mf_num_vector):
    """This function creates an initial Sugeno-type FIS
    for ANFIS training by performing
    a grid partiion on the given data. In this function, 
    the generalized bell-shaped function,
    i.e.,f(x;a,b,c) = 1/(1+((x-c)/a)^(2b)), 
    is adopted as the membership function of each input, and
    a linear weighed sum, i.e. sum_{i=1 to n}{wi*xi}+r, is adopted as the 
    membership function of a single output.
    
    Parameters
    ----------
    X: a 2-D array, shape (P,n) where P is the sample number and n is the 
    input number of each sample.
    Y: a 2-D array, shape(P,out_num) where P is the sampel number and 
    out_num is the output number.
    
    Returns
    ----------
    consequentParam: a list of 2-D arrays each with shape (N,n+1) given N rules and n inputs. This
    consequentParam is arranged as [[w_mi]], m=0,...,N-1; i=0,..,n
    premiseParam: a list of lists each containing the parameter arrays of membership
    functions, like [[1-D-array,...,1-D-array],...]
    
    the parameters set of the i-th node's j-th membership function is indexed by
    premiseParam[i][j], which is an array containing a, b, and c
    consequentParam[m,i] denotes the weight of the m-th rule's i-th input
    """
    # Generate the initial consequent parameters of the membership function of
    # the single output
    (P,n) = X.shape
    (PP,output_num) = Y.shape
    if len(mf_num_vector) == 0:
        mf_num_vector = [2 for i in range(n)]
        
    N = np.prod(mf_num_vector)
    consequentParam = [np.ones((N,n+1),dtype=float) for l in range(output_num)]


    # Generate the initial premise parameters
    premiseParam = []
    for i in range(n):
        # for the i-th input
        vmin = np.min(X[::,i])
        vmin = vmin*0.99
        vmax = np.max(X[::,i])
        vmax = vmax*1.01
        if mf_num_vector[i]==1:
            raise ValueError("The number of membership functions of the %d-th input should not be less than 1!" %
                             i+1)
        a = (vmax-vmin)/2/(mf_num_vector[i]-1)
        b = 2.0
        c_vector = np.linspace(vmin,vmax,num=mf_num_vector[i])
        #premiseParam_node_i = [np.array([a,b,c]) for c in c_vector]
        premiseParam.append([np.array([a,b,c]) for c in c_vector])
    return premiseParam, consequentParam
    
def index_of_2ndLayer(mf_num_vector):
    """This function generates the index of each node in the 2nd layer
    of the ANFIS.
    
    Parameters
    ----------
    mf_num_vector: a list like [m1,m2,...,mn] in which mi denotes the number of 
    membership functions w.r.t. i, i in {1,2,...,n}
    
    Returns
    ----------
    dict_index: a dictionary-type variable 
    like {(0,0,...,0):0,(0,0,...,1):1,..,(m1-1,m2-1,...,mn-1):N} where (j1,j2,..,jn)
    is a key, a tuple of indices of every node's membership functions, given that
    ji is the index of node i's j-th membership function and
    N = m1*m2*...*mn. Note that ji is ranging from 0 to mi.
    
    indices: a list of all the tuples each containing a combination of indices
    in the first layer, i.e., denoting a key in dict_index
    """
    I = []

    for mi in mf_num_vector:
        index = [ji for ji in range(mi)]
        I.append(index)
    indices = []
    indices = recursion(indices,0,I)
    #return {indices[value]:value for value in range(len(indices))}, indices
    return indices
    
    
def recursion(indices,i,I):
    """This function constructs a list of tuples of indices by using a recursion
    mechanism.
    
    Parameters
    ----------
    indices: a list of tuples like [(j1,j2,...,ji-1),...,(l1,l2,...,li-1)]
    
    Returns
    ----------
    indices: a list of all the tuples each containing a combination of indices
    """
    n = len(I)
    if len(indices)==0:
        indices = [(x,) for x in I[i]]
        return recursion(indices,i+1,I)
    elif i == (n-1):
        return [inds+(x,) for inds in indices for x in I[i]]
    else:
        indices = [inds+(x,) for inds in indices for x in I[i]]
        return recursion(indices,i+1,I)
        

def get_indices_of_2ndLayer(i,j,indices):
    """Given the index ji of a membership function in the 1st layer, indicated by (i,j),
    get the indices of the nodes in the 2nd layer that are connecting to this 
    membership function node.
    
    Parameters
    ----------
    i: an integer, ranging from 0 to n when considering there are n inputs
    j: an integer, ranging from 0 to mi where mi is the number of membership
    functions associated with the i-th input
    indices: a list of tuples each containing a combination of indices of all
    the membership functions
    
    Returns
    ----------
    m: a list containing the indices of the nodes in the 2nd layer that are 
    connecting to the i-th node's j-th membership function.
    """
    m = []
    for ind in range(len(indices)):
        key = indices[ind]
        if key[i] == j:
            m.append(ind)
    return m

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from prepare_data import load_test_data
    training_data,validation_data,all_data = load_test_data()
    # network initialization
    net = ANFIS([4,1])
    net.initialization(training_data)
    # train the net
    evaluation_cost, evaluation_accuracy, \
    training_cost, training_accuracy =net.stochastic_gradient_descent(training_data,
                                    validation_data,
                                    eta = 0.1,
                                    mini_batch_size=10,
                                    epoch = 30,
                                    adapt_eta_mode = True,
                                    evaluation_track_flag=True,
                                    training_track_flag=True)
    
    predictions = np.array([net.inference(x.reshape(-1,1))[0,0] for \
    (x,y) in validation_data])
    actual_outputs = np.array([y[0] for (x,y) in validation_data])
    labelfont = {"family":"serif","size":25}
    plt.figure(0,figsize=(9,8))
    plt.plot(actual_outputs,'-b',lw=8.0, label="Actual")
    plt.plot(predictions,'--r',lw=8.0, label="ANFIS")
    plt.grid(True)
    plt.legend(prop={"size":labelfont["size"],"family":"serif"})  
    xlabelstr = "Epoch $k$"
    ylabelstr = "Fuel level"
    plt.xlabel(xlabelstr,fontdict=labelfont)
    plt.ylabel(ylabelstr,fontdict=labelfont)
    plt.xticks(fontsize=labelfont["size"],\
    fontname=labelfont["family"])
    plt.yticks(fontsize=labelfont["size"],
               family=labelfont["family"])
    plt.show()