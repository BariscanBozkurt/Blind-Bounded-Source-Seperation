"""
Title: WBSS.py
Weighted Bounded Source Seperation Implementation
Reference: Alper T. Erdoğan and Cengiz Pehlevan, 'Blind Source Seperation Using Neural Networks with Local Learning Rules',ICASSP 2020

Code Writer: Barışcan Bozkurt (Koç University - EEE & Mathematics)

Date: 15.02.2021
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class OnlineWBSS:
    """
    Implementation of online one layer Weighted Bounded Source Seperation Recurrent Neural Network.
    Reference: Alper T. Erdoğan and Cengiz Pehlevan, 'Blind Source Seperation Using Neural Networks with Local Learning Rules',ICASSP 2020
    
    Parameters:
    =================================
    s_dim          -- Dimension of the sources
    x_dim          -- Dimension of the mixtures
    W              -- Initial guess for forward weight matrix W, must be size of s_dim by x_dim
    M              -- Initial guess for lateral weight matrix M, must be size of s_dim by s_dim
    D              -- Initial guess for weight (similarity weights) matrix, must be size of s_dim by s_dim
    gamma          -- Forgetting factor for data snapshot matrix
    mu, beta       -- Similarity weight update parameters, check equation (15) from the paper
    
    Methods:
    ==================================
    
    whiten_signal(X)        -- Whiten the given batch signal X
    
    ProjectOntoLInfty(X)   -- Project the given vector X onto L_infinity norm ball
    
    fit_next(x_online)     -- Updates the network parameters for one data point x_online
    
    fit_batch(X_batch)     -- Updates the network parameters for given batch data X_batch (but in online manner)
    
    """
    def __init__(self, s_dim, x_dim, gamma = 0.9999, mu = 1e-3, beta = 1e-7, W = None, M = None, D = None ):
        if W is not None:
            assert W.shape == (s_dim, x_dim), "The shape of the initial guess W must be (s_dim,x_dim)=(%d,%d)" % (s_dim, x_dim)
            W = W
        else:
            W = np.random.randn(s_dim,x_dim)
            W = (W / np.sqrt(np.sum(np.abs(W)**2,axis = 1)).reshape(s_dim,1))
            
        if M is not None:
            assert M.shape == (s_dim, s_dim), "The shape of the initial guess W must be (s_dim,s_dim)=(%d,%d)" % (s_dim, s_dim)
            M = M
        else:
            M = np.eye(s_dim)  
            
        if D is not None:
            assert D.shape == (s_dim, s_dim), "The shape of the initial guess W must be (s_dim,s_dim)=(%d,%d)" % (s_dim, s_dim)
            D = D
        else:
            D = 10*np.eye(s_dim)
            
        self.s_dim = s_dim
        self.x_dim = x_dim
        self.gamma = gamma
        self.mu = mu
        self.beta = beta
        self.W = W
        self.M = M
        self.D = D
        
    def whiten_signal(self, X, mean_normalize = True, type_ = 2):
        """
        Input : X  ---> Input signal to be whitened

        type_ : Defines the type for preprocesing matrix. type_ = 1 and 2 uses eigenvalue decomposition whereas type_ = 3 uses SVD.

        Output: X_white  ---> Whitened signal, i.e., X_white = W_pre @ X where W_pre = (R_x^0.5)^+ (square root of sample correlation matrix)
        """
        if mean_normalize:
            X = X - np.mean(X,axis = 0, keepdims = True)

        cov = np.cov(X.T)

        if type_ == 3: # Whitening using singular value decomposition
            U,S,V = np.linalg.svd(cov)
            d = np.diag(1.0 / np.sqrt(S))
            W_pre = np.dot(U, np.dot(d, U.T))

        else: # Whitening using eigenvalue decomposition
            d,S = np.linalg.eigh(cov)
            D = np.diag(d)

            D_sqrt = np.sqrt(D * (D>0))

            if type_ == 1: # Type defines how you want W_pre matrix to be
                W_pre = np.linalg.pinv(S@D_sqrt)
            elif type_ == 2:
                W_pre = np.linalg.pinv(S@D_sqrt@S.T)

        X_white = (W_pre @ X.T).T

        return X_white, W_pre
    
    def ProjectOntoLInfty(self, X):
        
        return X*(X>=-1.0)*(X<=1.0)+(X>1.0)*1.0-1.0*(X<-1.0)
    
    def fit_next(self, x_current):
        W = self.W
        M = self.M
        D = self.D
        gamma, mu, beta = self.gamma, self.mu, self.beta
        
        Upsilon = np.diag(np.diag(M))
        
        u = np.linalg.solve(M @ D, W @ x_current)
        y = self.ProjectOntoLInfty(u / np.diag(Upsilon * D))
        
        W = (gamma ** 2) * W + (1 - gamma ** 2) * np.outer(y,x_current)
        M = (gamma ** 2) * M + (1 - gamma ** 2) * np.outer(y,y)
        
        D = (1 - beta) * D + mu * np.diag(np.sum(np.abs(W)**2,axis = 1) - np.diag(M @ D @ M ))
        
        self.W = W
        self.M = M
        self.D = D
        
        return y
        
    def fit_batch(self, X, n_epochs = 10, whiten = True, shuffle = False, verbose = True):
        gamma, mu, beta, W, M, D = self.gamma, self.mu, self.beta, self.W, self.M, self.D
        
        assert X.shape[1] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[0]
        
        Y = np.zeros((samples, self.s_dim))
        
        
        if shuffle:
            idx = np.random.permutation(samples) # random permutation
        else:
            idx = np.arange(samples)
            
        if whiten:
            X_white, W_pre = self.whiten_signal(X)
        else:
            X_white = X
            
        if verbose:
            for k in range(n_epochs):
                for i_sample in tqdm(range(samples)):
                    x_current = X_white[idx[i_sample],:] # Take one input
                    Upsilon = np.diag(np.diag(M)) # Following paragraph of equation (16)
                    
                    # Neural Dynamics: Equations (17) from the paper
                    
                    u = np.linalg.solve(M @ D, W @ x_current)
                    y = self.ProjectOntoLInfty(u / np.diag(Upsilon * D))
                    
                    # Synaptic & Similarity weight updates, follows from equations (12,13,14,15,16) from the paper
                    
                    W = (gamma ** 2) * W + (1 - gamma ** 2) * np.outer(y,x_current)
                    M = (gamma ** 2) * M + (1 - gamma ** 2) * np.outer(y,y)
                    D = (1 - beta) * D + mu * np.diag(np.sum(np.abs(W)**2,axis = 1) - np.diag(M @ D @ M ))
                    
                    # Record the seperated signal
                    Y[idx[i_sample],:] = y
        else:
            for k in range(n_epochs):
                for i_sample in (range(samples)):
                    x_current = X_white[idx[i_sample],:] # Take one input
                    Upsilon = np.diag(np.diag(M)) # Following paragraph of equation (16)
                    
                    # Neural Dynamics: Equations (17) from the paper
                    
                    u = np.linalg.solve(M @ D, W @ x_current)
                    y = self.ProjectOntoLInfty(u / np.diag(Upsilon * D))
                    
                    # Synaptic & Similarity weight updates, follows from equations (12,13,14,15,16) from the paper
                    
                    W = (gamma ** 2) * W + (1 - gamma ** 2) * np.outer(y,x_current)
                    M = (gamma ** 2) * M + (1 - gamma ** 2) * np.outer(y,y)
                    D = (1 - beta) * D + mu * np.diag(np.sum(np.abs(W)**2,axis = 1) - np.diag(M @ D @ M ))
                    
                    # Record the seperated signal
                    Y[idx[i_sample],:] = y
        self.W = W
        self.M = M
        self.D = D
                    
        return Y

def whiten_signal(X, mean_normalize = True, type_ = 3):
    """
    Input : X  ---> Input signal to be whitened
    
    type_ : Defines the type for preprocesing matrix. type_ = 1 and 2 uses eigenvalue decomposition whereas type_ = 3 uses SVD.
    
    Output: X_white  ---> Whitened signal, i.e., X_white = W_pre @ X where W_pre = (R_x^0.5)^+ (square root of sample correlation matrix)
    """
    if mean_normalize:
        X = X - np.mean(X,axis = 0, keepdims = True)
    
    cov = np.cov(X.T)
    
    if type_ == 3: # Whitening using singular value decomposition
        U,S,V = np.linalg.svd(cov)
        d = np.diag(1.0 / np.sqrt(S))
        W_pre = np.dot(U, np.dot(d, U.T))
        
    else: # Whitening using eigenvalue decomposition
        d,S = np.linalg.eigh(cov)
        D = np.diag(d)

        D_sqrt = np.sqrt(D * (D>0))

        if type_ == 1: # Type defines how you want W_pre matrix to be
            W_pre = np.linalg.pinv(S@D_sqrt)
        elif type_ == 2:
            W_pre = np.linalg.pinv(S@D_sqrt@S.T)
    
    X_white = (W_pre @ X.T).T
    
    return X_white, W_pre

def ZeroOneNormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def ZeroOneNormalizeColumns(X):
    X_normalized = np.empty_like(X)
    for i in range(X.shape[1]):
        X_normalized[:,i] = ZeroOneNormalizeData(X[:,i])

    return X_normalized

def Subplot_gray_images(I, image_shape = [512,512], height = 15, width = 15, title = ''):
    
    n_images = I.shape[1]
    fig, ax = plt.subplots(1,n_images)
    fig.suptitle(title)
    fig.set_figheight(height)
    fig.set_figwidth(width)
    for i in range(n_images):
        ax[i].imshow(I[:,i].reshape(image_shape[0],image_shape[1]), cmap = 'gray')
    
    plt.show()
    