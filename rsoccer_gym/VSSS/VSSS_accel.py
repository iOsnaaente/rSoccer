from rSoccer.rsoccer_gym.Entities.PotentialField import _potential_core
from numba import njit

import numpy as np
import math


@njit( cache = True )
def _compute_grid_centers_core(half_length: float, half_width: float, spacing: float):
    # Determina número de linhas e colunas
    n_cols = int(math.floor((2*half_length) / spacing))
    n_rows = int(math.floor((2*half_width)  / spacing))
    # Cria um array vazio para armazenar os pontos ( x, y ) 
    centers = np.empty( (n_rows, n_cols, 2), dtype = np.float64)
    x0 = -half_length + spacing/2
    y0 = -half_width  + spacing/2
    for i in range(n_rows):
        yi = y0 + i*spacing
        for j in range(n_cols):
            xi = x0 + j*spacing
            centers[i,j,0] = xi
            centers[i,j,1] = yi
    return centers


@njit( cache = True )
def _compute_heatmap_core(
    centers: np.ndarray,
    ball_x: float, ball_y: float,
    ball_vx: float, ball_vy: float, 
    ball_theta: float, ball_v_theta: float,
    robots_x: np.ndarray, robots_y: np.ndarray,
    robots_vx: np.ndarray, robots_vy: np.ndarray,
    robots_theta: np.ndarray, robots_vtheta: np.ndarray,
    robots_A: np.ndarray, robots_sigma2: np.ndarray, robots_beta: np.ndarray,
    robots_epsilon: float,
    robots_gamma: np.ndarray, robots_kappa: np.ndarray, robots_omega_max: np.ndarray,
    robots_linear_max: np.ndarray, robots_k_stretch: np.ndarray,
    influence_radius2: float) -> np.ndarray:
    n_rows, n_cols, _ = centers.shape
    colors = np.empty((n_rows, n_cols, 4), dtype=np.uint8)
    
    # Para normalizar o campo entre [-1,1] 
    values = np.empty((n_rows, n_cols, 1), dtype=np.float64)
    max_value = 0.0 
    min_value = 0.0 

    # Determina o A global para normalização
    A_max = robots_A[0]

    # loop over grid
    for i in range(n_rows):
        for j in range(n_cols):
            U_total = 0.0
            # Sum potentials from all robots
            for r in range(robots_x.shape[0]):
                dx = centers[i,j,0] - robots_x[r]
                dy = centers[i,j,1] - robots_y[r]
                # Skip if outside this robot's influence
                if dx*dx + dy*dy > influence_radius2:
                    continue
                U = _potential_core(
                    dx, dy,
                    robots_vx[r], robots_vy[r], robots_theta[r], robots_vtheta[r],
                    robots_A[r], robots_sigma2[r], robots_beta[r],
                    robots_epsilon, robots_gamma[r], robots_kappa[r], robots_omega_max[r],
                    robots_k_stretch[r], robots_linear_max[r]
                )
                if U > 0.0:
                    U_total += U / A_max
                    
            # add ball potential
            dx = centers[i,j,0] - ball_x
            dy = centers[i,j,1] - ball_y
            # skip if outside ball's influence
            if dx*dx + dy*dy < influence_radius2:
                U = _potential_core(
                    dx, dy,
                    ball_vx, ball_vy,
                    ball_theta, ball_v_theta,
                    2.50, (0.1**2), 10.0,      # A, sigma2, beta
                    1e-5,                           # Epsilon 
                    1e-3, 25e-3,                    # Gamma e Kappa 
                    45,                             # Omega Max 
                    1.0, 1.0,                       # K_stretch e V_lin_max
                )
                if U > 0.0:
                    U_total -= U / A_max

            # Para normalizar
            values[i, j] = U_total 
            max_value = max_value if U_total <= max_value else U_total 
            min_value = min_value if U_total >= min_value else U_total 
    

    for i in range(n_rows):
        for j in range(n_cols):
            # Normaliza entre [ 0, Max_Value ]
            u = ( values[i,j,0] ) / max_value 

            # if u < -0.05:
            #     t = min(u, 1.0)  # t ∈ [0,1]
            #     r_ = 0
            #     g_ = 0
            #     b_ = 255
            #     a_ = int( 255 * (1-t) ) 
            if u <= 0.05:
                r_ = 0
                g_ = 0
                b_ = 0
                a_ = 0
            else:
                t = min(u, 1.0)  # t ∈ [0,1]
                r_ = 255
                g_ = 0 
                b_ = 0
                a_ = int( 255 * t ) 
            colors[i,j,0] = r_ & 0xFF
            colors[i,j,1] = g_ & 0xFF
            colors[i,j,2] = b_ & 0xFF
            colors[i,j,3] = a_ & 0xFF
    return colors
