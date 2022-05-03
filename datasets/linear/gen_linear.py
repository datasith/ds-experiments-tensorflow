import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.datasets import make_regression, make_classification

def create_poly(seed=42, n=100, noise=0.1, bias=-5, slope=2, feature_range=(0,5)):
    """
    m = 100                 # number of samples
    x = np.random.rand(m,1) # uniformly distributed random numbers
    theta_0 = 50            # intercept
    theta_1 = 35            # slope
    noise_sigma = 3
    noise = noise_sigma*np.random.randn(m,1) # gaussian random noise
    y = theta_0 + theta_1*x + noise          # noise added target
    """
    _data = np.linspace(feature_range[0],feature_range[1],n).reshape(n,-1)
    x = PolynomialFeatures(degree=1).fit_transform(_data)
    x[:,1:] = MinMaxScaler(feature_range=feature_range,copy=False).fit_transform(x[:,1:])
    data = x[:,1]
    np.random.seed(seed=seed)    
    noise = np.random.normal(0,noise,size=np.shape(data))
    y = slope*data+noise+bias
    y = y.reshape(n,1)
    return x, y

def create_regression(bias=20, random_state=1234):
    x, y = make_regression(1000, 2, n_informative=2, bias=bias, random_state=random_state)
    x, y = x.astype(np.float32), y.astype(np.float32).reshape(-1, 1)
    return x, y

def create_classification(rnd=1234):
    X, y = make_classification(n_samples=5000, n_features=2, n_redundant=0,
        n_informative=2, n_classes=2, random_state=rnd,
        n_clusters_per_class=1, class_sep=3, hypercube=True)
    return X, y