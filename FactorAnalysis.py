############################################################# 
## Stat 202A - Homework 5
## Author: 
## Date : 
## Description: This script implements factor analysis and 
## matrix completion
#############################################################

#############################################################
## INSTRUCTIONS: Please fill in the missing lines of code
## only where specified. Do not change function names, 
## function inputs or outputs. You can add examples at the
## end of the script (in the "Optional examples" section) to 
## double-check your work, but MAKE SURE TO COMMENT OUT ALL 
## OF YOUR EXAMPLES BEFORE SUBMITTING.
##
## Very important: Do not change the working directory anywhere
## in your code. If you do, I will be unable to grade your 
## work since Python will attempt to change my working directory
## to one that does not exist.
#############################################################

import numpy as np

def mySweep(A, m):
    """
    Perform a SWEEP operation on A with the pivot element A[m,m].
    
    :param A: a square matrix.
    :param m: the pivot element is A[m, m].
    :returns a swept matrix. Original matrix is unchanged.
    """
    
    ## No need to change anything here
    B = np.copy(A)   
    n = B.shape[0]
    for k in range(m):
        for i in range(n):
            for j in range(n):
                if i!=k and j!=k:
                    B[i,j] = B[i,j] - B[i,k]*B[k,j] / B[k,k]
        for i in range(n):
            if i!=k:
                 B[i,k] = B[i,k] / B[k,k]
        for j in range(n):
            if j!=k:
                B[k,j] = B[k,j] / B[k,k]
        B[k,k] = -1/B[k,k]
    
    return(B)

    
def factorAnalysis(n = 10, p = 5, d = 2, sigma = 1, nIter = 1000):
   
    """
    Perform factor analysis on simulated data.
    Simulate data X from the factor analysis model, e.g. 
    X = Z_true * W.T + epsilon
    where W_true is p * d loading matrix (numpy array), Z_true is a n * d matrix 
    (numpy array) of latent factors (assumed normal(0, I)), and epsilon is iid 
    normal(0, sigma^2) noise. You can assume that W_true is normal(0, I)
    
    :param n: Sample size.
    :param p: Number of variables
    :param d: Number of latent factors
    :param sigma: Standard deviation of noise
    :param nIter: Number of iterations
    """

    ## FILL CODE HERE
    
    W_true = np.random.standard_normal((p,d))
    Z_true = np.random.standard_normal((d,n))
    epsilon = sigma*np.random.standard_normal((p,n))

    X = W_true.dot(Z_true)+epsilon

    sq = 1.0
    XX = X.dot(X.T)

    W = (0.1)*np.random.standard_normal((p,d)) 
    
    for i in range(0,nIter):
       A = np.row_stack((np.column_stack(((W.T.dot(W))/sq+(np.eye(d)), W.T/sq)), np.column_stack((W/sq, (np.eye(p)))))) 
       AS = mySweep(A, d)
       alpha = AS[0:(d), d:(d+p)]
       D = -AS[0:(d), 0:(d)]
       Zh = alpha.dot(X) 
       ZZ = Zh.dot(Zh.T) + D*n 
       B = np.row_stack((np.column_stack((ZZ, Zh.dot(X.T))), np.column_stack((X.dot(Zh.T), XX))))
       BS = mySweep(B, d) 
       W = (BS[0:(d), d:(d+p)]).T 
       sq = np.mean(np.diag(BS[d:(d+p), d:(d+p)]))/n
 
    w = W
    ## Return the p * d np.array w, the estimate of the loading matrix
    return(w)
    
    
def matrixCompletion(n = 200, p = 100, d = 3, sigma = 0.1, nIter = 100,
                     prob = 0.2, lam = 0.1):
   
    """
    Perform matrix completion on simulated data.
    Simulate data X from the factor analysis model, e.g. 
    X = Z_true * W.T + epsilon
    where W_true is p * d loading matrix (numpy array), Z_true is a n * d matrix 
    (numpy array) of latent factors (assumed normal(0, I)), and epsilon is iid 
    normal(0, sigma^2) noise. You can assume that W_true is normal(0, I)
    
    :param n: Sample size.
    :param p: Number of variables
    :param d: Number of latent factors
    :param sigma: Standard deviation of noise
    :param nIter: Number of iterations
    :param prob: Probability that an entry of the matrix X is not missing
    :param lam: Regularization parameter
    """

    ## FILL CODE HERE
 
    W_true = np.random.standard_normal((p,d))
    Z_true = np.random.standard_normal((d,n))
    epsilon = sigma*np.random.standard_normal((p,n))

    X = W_true.dot(Z_true)+epsilon
    
    R = np.random.random_sample((p,n))

    for i in range(0,p):
        for j in range(0,n):
            if(R[i][j] < prob):
                R[i][j] = 1
            else:
                R[i][j] = 0
     
    W = np.random.standard_normal((p,d))*0.1
    Z = np.random.standard_normal((d,n))*0.1
   
    for it in range(0, nIter):
        for i in range(0,n):
            myrow, mycol = (np.diag(R[:,i])).shape
            WW = (W.T.dot(np.diag(R[:,i]))).dot(W) + lam*np.eye(d)
            WX = (W.T.dot(np.diag(R[:,i]))).dot(X[:,i].reshape(mycol,1))
            A = np.row_stack((np.column_stack((WW,WX)), np.column_stack((WX.T,0))))
            AS = mySweep(A,d)
            Z[:,i] = AS[0:d,d]

        for j in range(0,p):
            myrow, mycol = (np.diag(R[j,:])).shape
            ZZ = Z.dot(np.diag(R[j,:])).dot(Z.T) + lam*np.eye(d)
            ZX = Z.dot(np.diag(R[j,:])).dot(X[j,:].reshape(mycol,1))
            B = np.row_stack((np.column_stack((ZZ, ZX)), np.column_stack((ZX.T,0))))
            BS = mySweep(B, d)
            W[j,:] = BS[0:d, d]
    
    ## Return estimates of Z and W (both numpy arrays)
    return Z, W  
    
    
    

###########################################################
### Optional examples (comment out before submitting!!) ###
###########################################################
   
#print factorAnalysis()    
#Z,W = matrixCompletion()
#print Z,W
#print Z.shape,W.shape
