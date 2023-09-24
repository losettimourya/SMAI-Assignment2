import numpy as np
# from scipy import random
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import multivariate_normal

class GMM():
    def __init__(self, k, dim, init_mu=None, init_sigma=None, init_pi=None, colors=None):
        '''
        Define a model with a known number of clusters and dimensions.
        Args:
            - k: Number of Gaussian clusters
            - dim: Dimension 
            - init_mu: Initial value of mean of clusters (k, dim)
                       (default) random from uniform[-10, 10]
            - init_sigma: Initial value of covariance matrix of clusters (k, dim, dim)
                          (default) Identity matrix for each cluster
            - init_pi: Initial value of cluster weights (k,)
                       (default) equal value to all clusters i.e. 1/k
            - colors: Color value for plotting each cluster (k, 3)
                      (default) random from uniform[0, 1]
        '''
        self.k = k
        self.dim = dim
        if init_mu is None:
            init_mu = np.random.rand(k, dim) * 20 - 10
        self.mu = init_mu
        if init_sigma is None:
            init_sigma = np.zeros((k, dim, dim))
            for i in range(k):
                init_sigma[i] = np.eye(dim)
        self.sigma = init_sigma
        if init_pi is None:
            init_pi = np.ones(k) / k
        self.pi = init_pi
        if colors is None:
            colors = np.random.rand(k, 3)
        self.colors = colors
    
    def init_em(self, X):
        '''
        Initialization for EM algorithm.
        Args:
            - X: Data (batch_size, dim)
        '''
        self.data = X
        self.num_points = X.shape[0]
        self.z = np.zeros((self.num_points, self.k))
    
    def e_step(self):
        '''
        E-step of EM algorithm.
        '''
        for i in range(self.k):
            self.z[:, i] = self.pi[i] * self.mvn_pdf(self.data, self.mu[i], self.sigma[i])

        # Avoid NaN and division by zero by adding a small constant
        denominator = self.z.sum(axis=1, keepdims=True)
        denominator[denominator < np.finfo(float).eps] = np.finfo(float).eps
        self.z /= denominator

    
    def m_step(self):
        '''
        M-step of EM algorithm.
        '''
        sum_z = self.z.sum(axis=0)
        
        # Avoid NaN values by adding a small constant to prevent division by zero
        sum_z[sum_z < np.finfo(float).eps] = np.finfo(float).eps
        self.pi = sum_z / self.num_points
        self.mu = np.matmul(self.z.T, self.data)
        self.mu /= sum_z[:, None]
        for i in range(self.k):
            j = np.expand_dims(self.data, axis=1) - self.mu[i]
            s = np.matmul(j.transpose([0, 2, 1]), j)
            self.sigma[i] = np.matmul(s.transpose(1, 2, 0), self.z[:, i])
            
            # Add a small constant diagonal matrix to the covariance matrix to make it positive definite
            self.sigma[i] += np.eye(self.dim) * np.finfo(float).eps
            self.sigma[i] /= sum_z[i]
    
    def log_likelihood(self, X):
        '''
        Compute the log-likelihood of X under current parameters
        Args:
            - X: Data (batch_size, dim)
        Returns:
            - log-likelihood of X: Sum_n Sum_k log(pi_k * N( X_n | mu_k, sigma_k ))
        '''
        ll = []
        for d in X:
            tot = 0
            for i in range(self.k):
                # Regularized PDF calculation
                pdf = self.mvn_pdf(d, self.mu[i], self.sigma[i])
                if pdf > 0:
                    tot += self.pi[i] * pdf
            if tot > 0:
                ll.append(np.log(tot))
            else:
                # Handle cases where the PDFs are too close to zero
                ll.append(np.log(np.finfo(float).eps))
        return np.sum(ll)

    
    def mvn_pdf(self, x, mean, cov):
        '''
        Compute the probability density function of a multivariate normal distribution.
        Args:
            - x: Data point (dim,)
            - mean: Mean vector (dim,)
            - cov: Covariance matrix (dim, dim)
        Returns:
            - PDF value
        '''
        return multivariate_normal.pdf(x, mean=mean, cov=cov)


    
    def plot_gaussian(self, mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
        '''
        Utility function to plot one Gaussian from mean and covariance.
        '''
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0),
            width=ell_radius_x * 2,
            height=ell_radius_y * 2,
            facecolor=facecolor,
            **kwargs)
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = mean[0]
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = mean[1]
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    def draw(self, ax, n_std=2.0, facecolor='none', **kwargs):
        '''
        Function to draw the Gaussians.
        Note: Only for two-dimensionl dataset
        '''
        if(self.dim != 2):
            print("Drawing available only for 2D case.")
            return
        for i in range(self.k):
            self.plot_gaussian(self.mu[i], self.sigma[i], ax, n_std=n_std, edgecolor=self.colors[i], **kwargs)