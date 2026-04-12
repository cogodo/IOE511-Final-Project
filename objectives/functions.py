# All problems + associated gradients and Hessians
# IOE 511/MATH 562, University of Michigan
# Code written by: Albert S. Berahas & Jiahao Shi
# Modified by: Pearl Lin, Erick Vega, Colin Gordon

import os
import numpy as np
import scipy.io
import numpy.typing as npt
from einops import rearrange

Array = npt.NDArray[np.float64]

_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Define all the functions and calculate their gradients and Hessians, those functions include:
# (1)(2)(3)(4) Quadractic function
# (5)(6) Quartic function
# (7)(8) Rosenbrock function 
# (9) Data fit
# (10)(11) Exponential

 
# Problem Number: 1
# Problem Name: quad_10_10
# Problem Description: A randomly generated convex quadratic function; the 
#                      random seed is set so that the results are 
#                      reproducable. Dimension n = 10; Condition number
#                      kappa = 10

# function that computes the function value of the quad_10_10 function

def quad_10_10_func(x: Array) -> float:
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10,1))

    mat = scipy.io.loadmat(os.path.join(_DATA_DIR, 'quad_10_10_Q.mat'))
    Q = mat['Q']
    
    # compute function value
    return (1/2*x.T @ Q @ x + q.T @ x)[0]

def quad_10_10_grad(x: Array) -> Array:
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10,1))
    mat = scipy.io.loadmat(os.path.join(_DATA_DIR, 'quad_10_10_Q.mat'))
    Q = mat['Q']
    
    return Q @ x + q   
    

def quad_10_10_Hess(x: Array) -> Array:
    np.random.seed(0)
    mat = scipy.io.loadmat(os.path.join(_DATA_DIR, 'quad_10_10_Q.mat'))
    Q = mat['Q']
    
    return Q

# Problem Number: 2
# Problem Name: quad_10_1000
# Problem Description: A randomly generated convex quadratic function; the 
#                      random seed is set so that the results are 
#                      reproducable. Dimension n = 10; Condition number
#                      kappa = 1000

# function that computes the function value of the quad_10_1000 function

 

def quad_10_1000_func(x: Array) -> float:
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10,1))
    Q = scipy.io.loadmat(os.path.join(_DATA_DIR, 'quad_10_1000_Q.mat'))['Q']
    
    return float(0.5 * x.T @ Q @ x + q.T @ x)


def quad_10_1000_grad(x: Array) -> Array:
    np.random.seed(0)
    q = np.random.normal(size=(10,1))
    Q = scipy.io.loadmat(os.path.join(_DATA_DIR, 'quad_10_1000_Q.mat'))['Q']
    return Q @ x + q


def quad_10_1000_Hess(x: Array) -> Array:
    return scipy.io.loadmat(os.path.join(_DATA_DIR, 'quad_10_1000_Q.mat'))['Q']

# Problem Number: 3
# Problem Name: quad_1000_10
# Problem Description: A randomly generated convex quadratic function; the 
#                      random seed is set so that the results are 
#                      reproducable. Dimension n = 1000; Condition number
#                      kappa = 10

# function that computes the function value of the quad_1000_10 function

def quad_1000_10_func(x: Array) -> float:
    np.random.seed(0)
    q = np.random.normal(size=(1000,1))
    Q = scipy.io.loadmat(os.path.join(_DATA_DIR, 'quad_1000_10_Q.mat'))['Q']
    return float(0.5 * x.T @ Q @ x + q.T @ x)


def quad_1000_10_grad(x: Array) -> Array:
    np.random.seed(0)
    q = np.random.normal(size=(1000,1))
    Q = scipy.io.loadmat(os.path.join(_DATA_DIR, 'quad_1000_10_Q.mat'))['Q']
    return Q @ x + q


def quad_1000_10_Hess(x: Array) -> Array:
    return scipy.io.loadmat(os.path.join(_DATA_DIR, 'quad_1000_10_Q.mat'))['Q']

# Problem Number: 4
# Problem Name: quad_1000_1000
# Problem Description: A randomly generated convex quadratic function; the 
#                      random seed is set so that the results are 
#                      reproducable. Dimension n = 1000; Condition number
#                      kappa = 1000

# function that computes the function value of the quad_10_10 function

def quad_1000_1000_func(x: Array) -> float:
    np.random.seed(0)
    q = np.random.normal(size=(1000,1))
    Q = scipy.io.loadmat(os.path.join(_DATA_DIR, 'quad_1000_1000_Q.mat'))['Q']
    return float(0.5 * x.T @ Q @ x + q.T @ x)


def quad_1000_1000_grad(x: Array) -> Array:
    np.random.seed(0)
    q = np.random.normal(size=(1000,1))
    Q = scipy.io.loadmat(os.path.join(_DATA_DIR, 'quad_1000_1000_Q.mat'))['Q']
    return Q @ x + q


def quad_1000_1000_Hess(x: Array) -> Array:
    return scipy.io.loadmat(os.path.join(_DATA_DIR, 'quad_1000_1000_Q.mat'))['Q']


# Problem Number: 5
# Problem Name: quartic_1
# Problem Description: A quartic function. Dimension n = 4

# function that computes the function value of the quartic_1 function


def quartic_1_func(x: Array) -> float:
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e-4
    return float(0.5 * (x.T @ x) + sigma / 4 * (x.T @ Q @ x) ** 2)

def quartic_1_grad(x: Array) -> Array:
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e-4
    Qx = Q @ x
    return x + sigma * float(x.T @ Qx) * Qx

def quartic_1_Hess(x: Array) -> Array:
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e-4
    Qx = Q @ x
    return np.eye(4) + sigma * (2 * Qx @ Qx.T + float(x.T @ Q @ x) * Q)

# Problem Number: 6
# Problem Name: quartic_2
# Problem Description: A quartic function. Dimension n = 4

# function that computes the function value of the quartic_2 function


def quartic_2_func(x: Array) -> float:
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e4
    return float(0.5 * (x.T @ x) + sigma / 4 * (x.T @ Q @ x) ** 2)

def quartic_2_grad(x: Array) -> Array:
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e4
    Qx = Q @ x
    return x + sigma * float(x.T @ Qx) * Qx

def quartic_2_Hess(x: Array) -> Array:
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e4
    Qx = Q @ x
    return np.eye(4) + sigma * (2 * Qx @ Qx.T + float(x.T @ Q @ x) * Q)

# Problem Number: 7
# Problem Name: Rosenbrock_2
# Problem Description: Rosenbrock function. Dimension n = 2

# function that computes the function value of the Rosenbrock_2 problem
def rosen_2_func(x: Array) -> float:
    x = rearrange(x, '... -> (...)')
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


# function that computes the gradient of the Rosenbrock_2 problem
def rosen_2_grad(x: Array) -> Array:
    x = rearrange(x, '... -> (...)')
    return np.array(
        [
            [2 * (-1 + x[0] + 200 * x[0] ** 3 - 200 * x[0] * x[1])],
            [200 * (-x[0] ** 2 + x[1])],
        ],
        dtype=float,
    )


# function that computes the Hessian value of the Rosenbrock_2 problem
def rosen_2_Hess(x: Array) -> Array:
    x = rearrange(x, '... -> (...)')
    return np.array(
        [
            [1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]],
            [-400 * x[0], 200],
        ],
        dtype=float,
    )

# Problem Number: 8
# Problem Name: Rosenbrock_100
# Problem Description: Rosenbrock function. Dimension n = 100

# function that computes the function value of the Rosenbrock_100 problem
def rosen_100_func(x: Array) -> float:
    x = rearrange(x, '... -> (...)')
    f = 0
    for i in range(99):
        f = f + (1 - x[i]) ** 2 + 100 * (x[i + 1] - x[i] ** 2) ** 2
    return f

# function that computes the gradient value of the Rosenbrock_100 problem
def rosen_100_grad(x: Array) -> Array:
    x = rearrange(x, '... -> (...)')
    g = np.zeros(shape=(100, 1))
    g[0] = 2 * (-1 + x[0] + 200 * x[0] ** 3 - 200 * x[0] * x[1])
    for i in range(1, 99):
        g[i] = 2 * (-1 + x[i] + 200 * x[i] ** 3 - 200 * x[i] * x[i + 1]) + 200 * (x[i] - x[i - 1] ** 2)
    g[99] = 200 * (x[99] - x[98] ** 2)

    return g

# function that computes the Hessian value of the Rosenbrock_100 problem
def rosen_100_Hess(x: Array) -> Array:
    x = rearrange(x, '... -> (...)')
    H = np.zeros(shape=(100, 100))
    H[0, 0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
    H[0, 1] = -400 * x[0]
    for i in range(1, 99):
        H[i, i - 1] = -400 * x[i - 1]
        H[i, i] = 1200 * x[i] ** 2 - 400 * x[i + 1] + 202
        H[i, i + 1] = -400 * x[i]
    H[99, 98] = -400 * x[98]
    H[99, 99] = 200

    return H

# Problem Number: 9
# Problem Name: DataFit_2
# Problem Description: 3 dim non-linear least squares
def datafit_2_func(x: Array) -> float:
    x_flat = rearrange(x, '... -> (...)')
    y = np.array([1.5, 2.25, 2.625], dtype=float)
    powers  = np.array([1.0, 2.0, 3.0])
    
    temp = (y - x_flat[0] * (1 - x_flat[1] ** powers)) ** 2.0

    return np.sum(temp)

def datafit_2_grad(x: Array) -> Array:
    x_flat = rearrange(x, '... -> (...)')
    x1, x2 = x_flat[0], x_flat[1]
    y = np.array([1.5, 2.25, 2.625], dtype=float)
    powers = np.array([1.0, 2.0, 3.0])

    r = y - x1 * (1 - x2 ** powers)
    dr_dx1 = -(1 - x2 ** powers)
    dr_dx2 = x1 * powers * x2 ** (powers - 1)

    return np.array([
        [2 * np.sum(r * dr_dx1)],
        [2 * np.sum(r * dr_dx2)],
    ])

def datafit_2_Hess(x: Array) -> Array:
    x_flat = rearrange(x, '... -> (...)')
    x1, x2 = x_flat[0], x_flat[1]
    y = np.array([1.5, 2.25, 2.625], dtype=float)
    powers = np.array([1.0, 2.0, 3.0])

    r = y - x1 * (1 - x2 ** powers)
    dr_dx1 = -(1 - x2 ** powers)
    dr_dx2 = x1 * powers * x2 ** (powers - 1)

    d2r_dx2dx2 = x1 * powers * (powers - 1) * x2 ** (powers - 2)
    d2r_dx1dx2 = powers * x2 ** (powers - 1)

    H = np.zeros((2, 2))
    H[0, 0] = 2 * np.sum(dr_dx1 ** 2)
    H[0, 1] = 2 * np.sum(dr_dx1 * dr_dx2 + r * d2r_dx1dx2)
    H[1, 0] = H[0, 1]
    H[1, 1] = 2 * np.sum(dr_dx2 ** 2 + r * d2r_dx2dx2)
    return H

# Problem Number: 10
# Problem Name: Exponential_10
# Problem Description: f(x) = tanh(x1/2) + 0.1*exp(-x1) + sum_{i=2}^{n}(xi - 1)^4.
#                      Dimension n = 10
def exp_10_func(x: Array) -> float:
    x_flat = rearrange(x, '... -> (...)')
    assert x_flat.size == 10, f"exp_10_func expects n=10, got {x_flat.size}"
    z1 = float(x_flat[0])
    
    term1 = np.tanh(z1 / 2.0) + 0.1 * np.exp(-z1)
    term2 = float(np.sum((x_flat[1:] - 1.0) ** 4))

    return float(term1 + term2)


def exp_10_grad(x: Array) -> Array:
    x_flat = rearrange(x, '... -> (...)').astype(float)
    assert x_flat.size == 10, f"exp_10_grad expects n=10, got {x_flat.size}"
    grad = np.zeros_like(x_flat)
    z1 = x_flat[0]
    sech_sq = 1.0 / np.cosh(z1 / 2.0) ** 2
    grad[0] = 0.5 * sech_sq - 0.1 * np.exp(-z1)
    grad[1:] = 4.0 * (x_flat[1:] - 1.0) ** 3
    return grad[:, None]


def exp_10_Hess(x: Array) -> Array:
    x_flat = rearrange(x, '... -> (...)').astype(float)
    assert x_flat.size == 10, f"exp_10_Hess expects n=10, got {x_flat.size}"
    hess = np.zeros((10, 10), dtype=float)
    z1 = x_flat[0]
    sech_sq = 1.0 / np.cosh(z1 / 2.0) ** 2
    hess[0, 0] = -0.5 * sech_sq * np.tanh(z1 / 2.0) + 0.1 * np.exp(-z1)
    hess[1:, 1:] = np.diag(12.0 * (x_flat[1:] - 1.0) ** 2)
    return hess


# Problem Number: 11
# Problem Name: Exponential_1000
# Problem Description: f(x) = tanh(x1/2) + 0.1*exp(-x1) + sum_{i=2}^{n}(xi - 1)^4.
#                      Dimension n = 1000
def exp_1000_func(x: Array) -> float:
    x_flat = rearrange(x, '... -> (...)')
    assert x_flat.size == 1000, f"exp_1000_func expects n=1000, got {x_flat.size}"
    z1 = float(x_flat[0])
    
    term1 = np.tanh(z1 / 2.0) + 0.1 * np.exp(-z1)
    term2 = float(np.sum((x_flat[1:] - 1.0) ** 4))

    return float(term1 + term2)


def exp_1000_grad(x: Array) -> Array:
    x_flat = rearrange(x, '... -> (...)').astype(float)
    assert x_flat.size == 1000, f"exp_1000_grad expects n=1000, got {x_flat.size}"
    grad = np.zeros_like(x_flat)
    z1 = x_flat[0]
    sech_sq = 1.0 / np.cosh(z1 / 2.0) ** 2
    grad[0] = 0.5 * sech_sq - 0.1 * np.exp(-z1)
    grad[1:] = 4.0 * (x_flat[1:] - 1.0) ** 3
    return grad[:, None]


def exp_1000_Hess(x: Array) -> Array:
    x_flat = rearrange(x, '... -> (...)').astype(float)
    assert x_flat.size == 1000, f"exp_1000_Hess expects n=1000, got {x_flat.size}"
    hess = np.zeros((1000, 1000), dtype=float)
    z1 = x_flat[0]
    sech_sq = 1.0 / np.cosh(z1 / 2.0) ** 2
    hess[0, 0] = -0.5 * sech_sq * np.tanh(z1 / 2.0) + 0.1 * np.exp(-z1)
    hess[1:, 1:] = np.diag(12.0 * (x_flat[1:] - 1.0) ** 2)
    return hess


# Problem Number: 12
# Problem Name: genhumps_5
# Problem Description: A multi-dimensional problem with a lot of humps.
#                      This problem is from the well-known CUTEr test set.

# function that computes the function value of the genhumps_5 function

def genhumps_5_func(x: Array) -> float:
    x = rearrange(x, '... -> (...)')
    f = 0.0
    for i in range(4):
        f = f + np.sin(2*x[i])**2*np.sin(2*x[i+1])**2 + 0.05*(x[i]**2 + x[i+1]**2)
    return f

# function that computes the gradient of the genhumps_5 function

def genhumps_5_grad(x: Array) -> Array:
    x = rearrange(x, '... -> (...)')
    g = np.zeros((5, 1))
    g[0] = 4*np.sin(2*x[0])*np.cos(2*x[0]) * np.sin(2*x[1])**2 + 0.1*x[0]
    g[1] = 4*np.sin(2*x[1])*np.cos(2*x[1]) * (np.sin(2*x[0])**2 + np.sin(2*x[2])**2) + 0.2*x[1]
    g[2] = 4*np.sin(2*x[2])*np.cos(2*x[2]) * (np.sin(2*x[1])**2 + np.sin(2*x[3])**2) + 0.2*x[2]
    g[3] = 4*np.sin(2*x[3])*np.cos(2*x[3]) * (np.sin(2*x[2])**2 + np.sin(2*x[4])**2) + 0.2*x[3]
    g[4] = 4*np.sin(2*x[4])*np.cos(2*x[4]) * np.sin(2*x[3])**2 + 0.1*x[4]
    return g


# function that computes the Hessian of the genhumps_5 function
def genhumps_5_Hess(x: Array) -> Array:
    x = rearrange(x, '... -> (...)')
    H = np.zeros((5, 5))
    H[0,0] =  8 * np.sin(2*x[1])**2 * (np.cos(2*x[0])**2 - np.sin(2*x[0])**2) + 0.1
    H[0,1] = 16 * np.sin(2*x[0])*np.cos(2*x[0]) * np.sin(2*x[1])*np.cos(2*x[1])
    H[1,1] =  8 * (np.sin(2*x[0])**2 + np.sin(2*x[2])**2) * (np.cos(2*x[1])**2 - np.sin(2*x[1])**2) + 0.2
    H[1,2] = 16 * np.sin(2*x[1])*np.cos(2*x[1]) * np.sin(2*x[2])*np.cos(2*x[2])
    H[2,2] =  8 * (np.sin(2*x[1])**2 + np.sin(2*x[3])**2) * (np.cos(2*x[2])**2 - np.sin(2*x[2])**2) + 0.2
    H[2,3] = 16 * np.sin(2*x[2])*np.cos(2*x[2]) * np.sin(2*x[3])*np.cos(2*x[3])
    H[3,3] =  8 * (np.sin(2*x[2])**2 + np.sin(2*x[4])**2) * (np.cos(2*x[3])**2 - np.sin(2*x[3])**2) + 0.2
    H[3,4] = 16 * np.sin(2*x[3])*np.cos(2*x[3]) * np.sin(2*x[4])*np.cos(2*x[4])
    H[4,4] =  8 * np.sin(2*x[3])**2 * (np.cos(2*x[4])**2 - np.sin(2*x[4])**2) + 0.1
    H[1,0] = H[0,1]
    H[2,1] = H[1,2]
    H[3,2] = H[2,3]
    H[4,3] = H[3,4]
    return H