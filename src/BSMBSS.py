"""
Title: WBSS.py
Weighted Bounded Source Seperation Implementation
Reference: Alper T. Erdoğan and Cengiz Pehlevan, 'Blind Source Seperation Using Neural Networks with Local Learning Rules',ICASSP 2020

Code Writer: Barışcan Bozkurt (Koç University - EEE & Mathematics)

Date: 11.02.2021
"""

import numpy as np
import scipy
from scipy.stats import invgamma, chi2, t
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
import matplotlib as mpl
import pylab as pl
from tqdm import tqdm
from numba import njit, jit
import math
from IPython.display import display, clear_output, Latex, Math
from IPython import display as display1

class OnlineBSM:
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
    def __init__(self, s_dim, x_dim, gamma = 0.9999, mu = 1e-3, beta = 1e-7, W = None, M = None, D = None, neural_OUTPUT_COMP_TOL = 1e-6, set_ground_truth = False, S = None, A = None):
        if W is not None:
            assert W.shape == (s_dim, x_dim), "The shape of the initial guess W must be (s_dim,x_dim)=(%d,%d)" % (s_dim, x_dim)
            W = W
        else:
            W = np.random.randn(s_dim,x_dim)
            W = 0.0033 * (W / np.sqrt(np.sum(np.abs(W)**2,axis = 1)).reshape(s_dim,1))
            # for k in range(W_HX.shape[0]):
            #     W_HX[k,:] = WScalings[0] * W_HX[k,:]/np.linalg.norm(W_HX[k,:])
            
        if M is not None:
            assert M.shape == (s_dim, s_dim), "The shape of the initial guess W must be (s_dim,s_dim)=(%d,%d)" % (s_dim, s_dim)
            M = M
        else:
            M = 0.02*np.eye(s_dim)  
            
        if D is not None:
            assert D.shape == (s_dim, s_dim), "The shape of the initial guess W must be (s_dim,s_dim)=(%d,%d)" % (s_dim, s_dim)
            D = D
        else:
            D = 1*np.eye(s_dim)
            
        self.s_dim = s_dim
        self.x_dim = x_dim
        self.gamma = gamma
        self.mu = mu
        self.beta = beta
        self.W = W
        self.M = M
        self.D = D
        self.neural_OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL
        self.set_ground_truth = set_ground_truth
        self.S = S
        self.A = A
        self.SIRlist = []
        
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

    # Calculate SIR Function
    def CalculateSIR(self, H,pH, return_db = True):
        G=pH@H
        Gmax=np.diag(np.max(abs(G),axis=1))
        P=1.0*((np.linalg.inv((Gmax))@np.abs(G))>0.99)
        T=G@P.T
        rankP=np.linalg.matrix_rank(P)
        diagT = np.diag(T)
        # Signal Power
        sigpow = np.linalg.norm(diagT,2)**2
        # Interference Power
        intpow = np.linalg.norm(T,'fro')**2 - sigpow
        SIRV = sigpow/intpow
        # SIRV=np.linalg.norm((np.diag(T)))**2/(np.linalg.norm(T,'fro')**2-np.linalg.norm(np.diag(T))**2)
        if return_db:
            SIRV = 10*np.log10(sigpow/intpow)

        return SIRV,rankP
    
    def ProjectOntoLInfty(self, X):

        return X*(X>=-1.0)*(X<=1.0)+(X>1.0)*1.0-1.0*(X<-1.0)
    
    def compute_overall_mapping(self,return_mapping = True):
        W, M, D = self.W, self.M, self.D

        Wf = np.linalg.pinv(M @ D) @ W
        self.Wf = Wf

        if return_mapping:
            return Wf
        else:
            return None

    @staticmethod
    @njit
    def run_neural_dynamics_antisparse(x, y, W, M, D,neural_dynamic_iterations = 250, lr_start = 0.1, lr_stop = 1e-3, tol = 1e-6, fast_start = False):

        def ProjectOntoLInfty(X, thresh = 1.0):
            return X*(X>=-thresh)*(X<=thresh)+(X>thresh)*thresh-thresh*(X<-thresh)
        
        Upsilon = np.diag(np.diag(M))
        M_hat = M - Upsilon
        u = Upsilon @ D @ y

        if fast_start:
            u = 0.99*np.linalg.solve(M @ D, W @ x)
            y = ProjectOntoLInfty(u / np.diag(Upsilon * D))

        for j in range(neural_dynamic_iterations):
            lr = max(lr_start/(1 + j), lr_stop)
            yold = y
            du = -u + (W @ x - M_hat @ D @ y)
            u = u - lr * du

            y = ProjectOntoLInfty(u / np.diag(Upsilon * D))

            if np.linalg.norm(y - yold) < tol * np.linalg.norm(y):
                break

        return y

    def fit_next_antisparse(self, x_current, neural_dynamic_iterations = 250, neural_lr_start = 0.3, neural_lr_stop = 1e-3, fast_start = False):
        W = self.W
        M = self.M
        D = self.D
        gamma, mu, beta = self.gamma, self.mu, self.beta
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        # Upsilon = np.diag(np.diag(M))
        
        # u = np.linalg.solve(M @ D, W @ x_current)
        # y = self.ProjectOntoLInfty(u / np.diag(Upsilon * D))
        y = np.random.randn(self.s_dim,)
        y = self.run_neural_dynamics_antisparse(x_current, y, W, M, D, neural_dynamic_iterations, neural_lr_start, neural_lr_stop, neural_OUTPUT_COMP_TOL, fast_start)

        
        W = (gamma ** 2) * W + (1 - gamma ** 2) * np.outer(y,x_current)
        M = (gamma ** 2) * M + (1 - gamma ** 2) * np.outer(y,y)
        
        D = (1 - beta) * D + mu * np.diag(np.sum(np.abs(W)**2,axis = 1) - np.diag(M @ D @ M ))
        
        self.W = W
        self.M = M
        self.D = D
        
        
    def fit_batch_antisparse(self, X, n_epochs = 1, whiten = False, shuffle = False, neural_dynamic_iterations = 250, neural_lr_start = 0.3, neural_lr_stop = 1e-3, fast_start = False, debug_iteration_point = 1000, plot_in_jupyter = False):
        gamma, mu, beta, W, M, D = self.gamma, self.mu, self.beta, self.W, self.M, self.D
        neural_OUTPUT_COMP_TOL = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth
        SIRlist = self.SIRlist

        if debugging:
            S = self.S
            A = self.A

        assert X.shape[0] == self.x_dim, "You must input the transpose"
        
        samples = X.shape[1]
        
        Y = np.zeros((self.s_dim, samples))
        Y = 0.05*np.random.randn(self.s_dim, samples)
        
        if shuffle:
            idx = np.random.permutation(samples) # random permutation
        else:
            idx = np.arange(samples)
            
        if whiten:
            X_white, W_pre = self.whiten_signal(X.T)
            X_white = X_white.T
            A = W_pre @ A
            self.A = A
        else:
            X_white = X
            

        for k in range(n_epochs):
            for i_sample in tqdm(range(samples)):
                x_current = X_white[:, idx[i_sample]] # Take one input
                y = Y[:, idx[i_sample]]

                # Upsilon = np.diag(np.diag(M)) # Following paragraph of equation (16)
                
                # Neural Dynamics: Equations (17) from the paper
                
                # u = np.linalg.solve(M @ D, W @ x_current)
                # y = self.ProjectOntoLInfty(u / np.diag(Upsilon * D))
                y = self.run_neural_dynamics_antisparse(x_current, y, W, M, D, neural_dynamic_iterations, neural_lr_start, neural_lr_stop, neural_OUTPUT_COMP_TOL, fast_start)
                
                # Synaptic & Similarity weight updates, follows from equations (12,13,14,15,16) from the paper
                
                W = (gamma ** 2) * W + (1 - gamma ** 2) * np.outer(y,x_current)
                M = (gamma ** 2) * M + (1 - gamma ** 2) * np.outer(y,y)
                D = (1 - beta) * D + mu * np.diag(np.sum(np.abs(W)**2,axis = 1) - np.diag(M @ D @ M ))
                
                # Record the seperated signal
                Y[:, idx[i_sample]] = y
                if debugging:
                    if (i_sample % debug_iteration_point) == 0:
                        self.W = W
                        self.M = M
                        self.D = D
                        Wf = self.compute_overall_mapping(return_mapping = True)
                        SIR,_ = self.CalculateSIR(A, Wf)
                        SIRlist.append(SIR)
                        self.SIRlist = SIRlist

                        if plot_in_jupyter:
                            pl.clf()
                            pl.plot(np.array(SIRlist), linewidth = 3)
                            pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize = 15)
                            pl.ylabel("SIR (dB)", fontsize = 15)
                            pl.title("SIR Behaviour", fontsize = 15)
                            pl.grid()
                            clear_output(wait=True)
                            display(pl.gcf())         

        self.W = W
        self.M = M
        self.D = D

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
    
def generate_correlated_uniform_sources(R, range_ = [-1,1], n_sources = 5, size_sources = 500000):
    """
    R : correlation matrix
    """
    assert R.shape[0] == n_sources, "The shape of correlation matrix must be equal to the number of sources, which is entered as (%d)" % (n_sources)
    S = np.random.uniform(range_[0], range_[1], size = (n_sources, size_sources))
    L = np.linalg.cholesky(R)
    S_ = L @ S
    return S_

def generate_correlated_copula_sources(rho = 0.0, df = 4, n_sources = 5, size_sources = 500000, decreasing_correlation = True):
    """
    rho     : correlation parameter
    df      : degrees for freedom

    required libraries:
    from scipy.stats import invgamma, chi2, t
    from scipy import linalg
    import numpy as np
    """
    if decreasing_correlation:
        first_row = np.array([rho ** j for j in range(n_sources)])
        calib_correl_matrix = linalg.toeplitz(first_row, first_row)
    else:
        calib_correl_matrix = np.eye(n_sources) * (1 - rho) + np.ones((n_sources, n_sources)) * rho

    mu = np.zeros(len(calib_correl_matrix))
    s = chi2.rvs(df, size = size_sources)[:, np.newaxis]
    Z = np.random.multivariate_normal(mu, calib_correl_matrix, size_sources)
    X = np.sqrt(df/s) * Z # chi-square method
    S = t.cdf(X, df).T
    return S

def display_matrix(array):
    data = ''
    for line in array:
        if len(line) == 1:
            data += ' %.3f &' % line + r' \\\n'
            continue
        for element in line:
            data += ' %.3f &' % element
        data += r' \\' + '\n'
    display(Math('\\begin{bmatrix} \n%s\end{bmatrix}' % data))

# Calculate SIR Function
def CalculateSIR(H,pH, return_db = True):
    G=pH@H
    Gmax=np.diag(np.max(abs(G),axis=1))
    P=1.0*((np.linalg.inv((Gmax))@np.abs(G))>0.99)
    T=G@P.T
    rankP=np.linalg.matrix_rank(P)
    diagT = np.diag(T)
    # Signal Power
    sigpow = np.linalg.norm(diagT,2)**2
    # Interference Power
    intpow = np.linalg.norm(T,'fro')**2 - sigpow
    SIRV = sigpow/intpow
    # SIRV=np.linalg.norm((np.diag(T)))**2/(np.linalg.norm(T,'fro')**2-np.linalg.norm(np.diag(T))**2)
    if return_db:
        SIRV = 10*np.log10(sigpow/intpow)

    return SIRV,rankP

def subplot_1D_signals(X, title = '',title_fontsize = 20, figsize = (10,5), linewidth = 1, colorcode = '#050C12'):
    """
    Plot the 1D signals (each column from the given matrix)
    """
    n = X.shape[1] # Number of signals
    
    fig, ax = plt.subplots(n,1, figsize = figsize)
    
    for i in range(n):
        ax[i].plot(X[:,i], linewidth = linewidth, color = colorcode)
        ax[i].grid()
    
    plt.suptitle(title, fontsize = title_fontsize)
    # plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
    plt.draw()

def plot_convergence_plot(metric, xlabel = '', ylabel = '', title = '', figsize = (12,8), fontsize = 15, linewidth = 3, colorcode = '#050C12'):
    
    plt.figure(figsize = figsize)
    plt.plot(metric, linewidth = linewidth, color = colorcode)
    plt.xlabel(xlabel, fontsize = fontsize)
    plt.ylabel(ylabel, fontsize = fontsize)
    plt.title(title, fontsize = fontsize)
    # plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
    plt.grid()
    plt.draw()
    
def find_permutation_between_source_and_estimation(S,Y):
    """
    S    : Original source matrix
    Y    : Matrix of estimations of sources (after BSS or ICA algorithm)
    
    return the permutation of the source seperation algorithm
    """
    
    # perm = np.argmax(np.abs(np.corrcoef(S.T,Y.T) - np.eye(2*S.shape[1])),axis = 0)[S.shape[1]:]
    perm = np.argmax(np.abs(np.corrcoef(Y.T,S.T) - np.eye(2*S.shape[1])),axis = 0)[S.shape[1]:]
    return perm

def signed_and_permutation_corrected_sources(S,Y):
    perm = find_permutation_between_source_and_estimation(S,Y)
    return np.sign((Y[:,perm] * S).sum(axis = 0)) * Y[:,perm]
