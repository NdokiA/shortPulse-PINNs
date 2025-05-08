import torch 
import torch.nn as nn
import numpy as np 
import json
from ssfmPack import utils

#Activation Function
    
def find_activation(activation):
    act_fn = {
    'tanh': nn.Tanh,
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'swish': nn.SiLU,
    }.get(activation.lower(), nn.Tanh)
    
    return act_fn 

#Optimizers 
def choose_optimizer(optimizer_name: str, *args, **kwargs):
    if optimizer_name.lower() == 'lbfgs':
        return LBFGS(*args, **kwargs)
    elif optimizer_name.lower() == 'adam':
        return Adam(*args, **kwargs)

def LBFGS(model_param,
        lr=1.0,
        max_iter=10000,
        max_eval=None,
        history_size=50,
        tolerance_grad=1e-20,
        tolerance_change=1e-20,
        line_search_fn="strong_wolfe"):

    optimizer = torch.optim.LBFGS(
        model_param,
        lr=lr,
        max_iter=max_iter,
        max_eval=max_eval,
        history_size=history_size,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        line_search_fn=line_search_fn
        )

    return optimizer

def Adam(model_param, lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False):

    optimizer = torch.optim.Adam(
                model_param,
                lr=lr,
    )
    return optimizer

#Logging Information on Dict.
class DictLogger:
    def __init__(self):
        self.history = {}
    
    def add(self, log_dict):
        for key, value in log_dict.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def save(self, path):
        history  = {k: [float(v) for v in vs] for k, vs in self.history.items()}
        with open(path, 'w') as f:
            json.dump(history, f, indent = 4)

#Sampling 
def lhs_sampling(n: int, d: int, seed: int = 0) -> np.ndarray:
        """
        Latin Hypercube Sampling (LHS) to generate random points

        Args:
            n (int): Number of samples
            d (int): Dimension of samples
            seed (int, optional): Random seed. Defaults to None

        Returns:
            np.ndarray: Random samples
        """        
        rng = np.random.default_rng(seed)
        result = np.zeros((n,d))
        
        for i in range(d):
            result[:,i] = rng.permutation(np.linspace(0,1,n,endpoint=False)) + rng.random((n))/n
        
        return result


def generate_points(start: np.ndarray, final: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    '''
    Initialize points using Latin Hypercube Sampling

    Args:
        final (np.ndarray): Expected final point (have to have the same dimension with initial)
        initial (np.ndarray): Expected initial point (have to have the same dimension with final)
        n (int): Number of samples
        seed (int, optional): Random seed. Defaults to None.
    '''
    d = final.shape[-1] if isinstance(final, np.ndarray) else 1
    points = lhs_sampling(n,d,seed)*(final-start) + start
    return points

def ssfm_sampling(timeArray, lengthArray, pulse, num_sample = 2000, clipRange = 500, normalized = True):
    clipTime = utils.clipMatrix(timeArray, clipRange)
    pulse = utils.clipMatrix(pulse, clipRange)
    T,L = np.meshgrid(clipTime, lengthArray) 

    samples =generate_points(np.array([0,0]), np.array([1,1]), num_sample)*pulse.shape
    samples = samples.astype(int)

    Ts = T[samples[:,0], samples[:,1]].reshape(-1,1)
    Ls = L[samples[:,0], samples[:,1]].reshape(-1,1)
    ps = pulse[samples[:,0], samples[:,1]].reshape(-1,1)
    if normalized:
        Ts = Ts/max(timeArray)
        Ls = Ls/max(lengthArray)
    
    us = np.real(ps)
    vs = np.imag(ps)

    tx = (Ts, Ls)
    uv = (us, vs)
    return tx, uv