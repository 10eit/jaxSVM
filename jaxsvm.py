import jax.numpy as jnp
from typing import Callable

def linear_kernel(X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
    K = X @ Y.transpose()
    return K

def rbf_kernel(X: jnp.ndarray, Y: jnp.ndarray, gamma: float) -> jnp.ndarray:
    X_norm = jnp.sum(X**2, axis=1)[:, None]  # shape (n,1)
    Y_norm = jnp.sum(Y**2, axis=1)[None, :]  # shape (1,m)
    K = jnp.exp(-gamma * (X_norm + Y_norm - 2*X @ Y.T))
    return K

def poly_kernel(X: jnp.ndarray, Y: jnp.ndarray, degree: int, coef0: float) -> jnp.ndarray:
    K = (X @ Y.T + coef0) ** degree
    return K


def gram_matrix(X: jnp.ndarray, kernel_fn: Callable, **kernel_params) -> jnp.ndarray:
    K = kernel_fn(X, X, **kernel_params)
    return K