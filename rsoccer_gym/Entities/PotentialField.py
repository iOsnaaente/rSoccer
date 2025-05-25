import math
from numba import njit

TEST_A = True
TEST_B = False


@njit( cache = True )
def _potential_core(dx: float,
                    dy: float,
                    v_x: float,
                    v_y: float,
                    theta_deg: float,
                    v_theta: float,
                    A: float,
                    sigma2: float,
                    beta: float,
                    epsilon: float,
                    gamma: float,
                    kappa: float,
                    omega_max: float) -> float:

    if TEST_A:
        """
        U(x,y) = A * exp(-d^2/(2σ^2)) * [1 + β * (v · e)],
        onde d = ||r - r_i|| e e = (r - r_i)/d.
        """
        d2 = dx*dx + dy*dy
        d  = math.sqrt( d2 + epsilon)
        # vetor unitário do robô para o ponto (x,y)
        ex, ey = dx/d, dy/d
        # componente de velocidade na direção e
        vdot = v_x * ex + v_y * ey
        return A * math.exp(-d2 / (2 * sigma2)) * (1 + beta * vdot)

    elif TEST_B:
        # 1) Normalização radial
        d2 = dx*dx + dy*dy
        d = math.sqrt(d2 + epsilon)
        inv_d = 1.0 / d
        ex = dx * inv_d
        ey = dy * inv_d

        # 2) Potencial base (gaussiana + velocidade linear)
        vdot = v_x*ex + v_y*ey
        U0 = A * math.exp(-d2 / (2.0 * sigma2)) * (1.0 + beta * vdot)

        # 3) Heading e direção da velocidade
        theta_rad = math.radians(theta_deg)
        hx = math.cos(theta_rad)
        hy = math.sin(theta_rad)
        v_lin = math.hypot(v_x, v_y)
        if v_lin > 1e-6:
            inv_vlin = 1.0 / v_lin
            vdir_x = v_x * inv_vlin
            vdir_y = v_y * inv_vlin
        else:
            vdir_x, vdir_y = hx, hy

        # 4) Blend heading ↔ vdir via alpha(omega)
        omega = abs(v_theta)
        alpha = omega / omega_max if omega < omega_max else 1.0
        bx = (1.0 - alpha)*hx + alpha*vdir_x
        by = (1.0 - alpha)*hy + alpha*vdir_y
        inv_bnorm = 1.0 / math.hypot(bx, by)
        dxn = bx * inv_bnorm
        dyn = by * inv_bnorm

        # 5) Máscara traseira e fator de orientação
        cos_main = dxn*ex + dyn*ey
        if cos_main <= 0.0:
            return 0.0
        dir_factor = 1.0 + gamma * cos_main

        # 6) Puxão lateral
        px = -hy
        py = hx
        turn_sign = 1.0 if v_theta > 0.0 else -1.0 if v_theta < 0.0 else 0.0
        cos_turn = (px*ex + py*ey) * turn_sign
        turn_factor = 1.0 + kappa * omega * cos_turn

        return U0 * dir_factor * turn_factor





@njit(cache=True)
def _gradient_core(dx: float,
                   dy: float,
                   v_x: float,
                   v_y: float,
                   A: float,
                   sigma: float,
                   beta: float,
                   epsilon: float) -> (float, float):
    d2 = dx*dx + dy*dy
    d = math.sqrt(d2 + epsilon)
    ex = dx / d
    ey = dy / d

    vdot = v_x * ex + v_y * ey
    exp_term = math.exp(-d2 / (2 * sigma * sigma))

    # Termo radial
    radial = - (1 + beta * vdot) * (d / (sigma * sigma))
    # Termo tangencial
    tang = beta / d
    tang_x = tang * (v_x - vdot * ex)
    tang_y = tang * (v_y - vdot * ey)

    dUdx = A * exp_term * (radial * ex + tang_x)
    dUdy = A * exp_term * (radial * ey + tang_y)
    return dUdx, dUdy


class PotentialField:
    def __init__(self, owner_robot,
                 A: float = 1.0,
                 sigma: float = 1.0,
                 beta: float = 1.0,
                 epsilon: float = 1e-6,
                 gamma: float = 0.1,
                 kappa: float = 0.1,
                 omega_max: float = 30.0):
        self.owner = owner_robot
        self.A = A
        self.sigma = sigma
        self.beta = beta
        self.epsilon = epsilon
        self.gamma = gamma
        self.kappa = kappa
        self.omega_max = omega_max
        self.sigma2 = sigma * sigma
        # Define raio de influência (padrão 3*sigma)
        self.influence_radius2 = (3.0 * self.sigma)**2

    def potential_at( self, x: float, y: float ) -> float:
        # 0) Checa se está dentro do raio de influência
        dx = x - self.owner.x
        dy = y - self.owner.y
        if ((dx*dx) + (dy*dy)) > self.influence_radius2:
            return 0.0
        # 1) Calcula potencial via função jitted
        return _potential_core(
            dx, dy,
            self.owner.v_x, self.owner.v_y,
            self.owner.theta, self.owner.v_theta,
            self.A, self.sigma2, self.beta,
            self.epsilon,
            self.gamma, self.kappa,
            self.omega_max
        )

    def gradient_at(self, x: float, y: float) -> tuple[float, float]:
        dx = x - self.owner.x
        dy = y - self.owner.y
        return _gradient_core(
            dx, dy,
            self.owner.v_x, self.owner.v_y,
            self.A, self.sigma, self.beta,
            self.epsilon
        )

    def force_at(self, x: float, y: float) -> tuple[float, float]:
        dUdx, dUdy = self.gradient_at(x, y)
        return -dUdx, -dUdy

    def influence_on(self, other_robot) -> tuple[float, float]:
        return self.force_at(other_robot.x, other_robot.y)

    def update_parameters(self,
                          A: float = None,
                          sigma: float = None,
                          beta: float = None,
                          epsilon: float = None,
                          gamma: float = None,
                          kappa: float = None,
                          omega_max: float = None):
        if A is not None:        self.A = A
        if beta is not None:     self.beta = beta
        if epsilon is not None:  self.epsilon = epsilon
        if gamma is not None:    self.gamma = gamma
        if kappa is not None:    self.kappa = kappa
        if omega_max is not None: self.omega_max = omega_max
        if sigma is not None:
            self.sigma = sigma
            self.sigma2 = sigma * sigma


# import numpy as np
# import math 


# class PotentialField:
#     def __init__( self, owner_robot: 'Robot',
#         A: float = 1.0,
#         sigma: float = 1.0,
#         beta: float = 1.0,
#         epsilon: float = 1e-6,
#         gamma: float = 0.1,
#         kappa: float = 0.1,
#         omega_max: float = 30.0
#     ):
#         self.owner      = owner_robot   # O Robot dono deste campo
#         self.epsilon    = epsilon       # Para evitar divisão por zero
#         self.sigma      = sigma         # Raio de influência (desvio padrão)
#         self.beta       = beta          # Peso do termo de velocidade
#         self.A          = A             # Amplitude máxima do potencial
#         self.gamma      = gamma         # Peso do bump na direção dinâmica
#         self.kappa      = kappa         # Ganho do puxão lateral
#         self.omega_max  = omega_max     # Escala para cálculo de alpha (velocidade angular)
#         self.sigma2     = sigma*sigma

#     def potential_at(self, x: float, y: float) -> float:
#         # """
#         # U(x,y) = A * exp(-d^2/(2σ^2)) * [1 + β * (v · e)],
#         # onde d = ||r - r_i|| e e = (r - r_i)/d.
#         # """
#         # dx = x - self.owner._x
#         # dy = y - self.owner._y
#         # d2 = dx*dx + dy*dy
#         # d  = np.sqrt(d2 + self.epsilon)

#         # # vetor unitário do robô para o ponto (x,y)
#         # ex, ey = dx/d, dy/d

#         # # componente de velocidade na direção e
#         # vdot = self.owner.v_x * ex + self.owner.v_y * ey
#         # return self.A * np.exp(-d2 / (2 * self.sigma**2)) * (1 + self.beta * vdot)
        
#         """
#         Potencial U(x,y) = U₀(x,y) · F_dir(x,y) · F_turn(x,y)
#         Onde:
#         U₀(x,y) = A · exp(-d² / (2·σ²)) · [1 + β · (v · e)]
#             • d   = distância do robô ao ponto = √(dx² + dy² + ε)
#             • e   = vetor unitário robo→ponto
#             • v·e = projeção da velocidade linear no radial
#         F_dir(x,y) = 1 + γ · (d̂ · e)
#             • d̂ = vetor de responsividade “dinâmico”, mistura entre
#                 - h = (cos θ, sin θ) (heading do robô) e
#                 - v̂ = direção da velocidade linear
#             por um peso α = min(|ω|/ωₘₐₓ, 1)
#             • γ controla quão forte o campo “puxa” para essa direção
#         F_turn(x,y) = 1 + κ · |ω| · [sign(ω)·(p · e)]
#             • p = vetor perpendicular a h (lado esquerdo)
#             • sign(ω) escolhe esquerda (+) ou direita (-)
#             • κ controla a intensidade do “puxão” lateral
#         Se (d̂·e) < 0, retornamos 0 para eliminar todo campo atrás do robô.

#         **Variáveis de tuning disponíveis:**
#         • A             — amplitude do potencial gaussiano  
#         • σ (sigma)     — largura do “bump” gaussiano  
#         • β (beta)      — ganho do termo de velocidade linear  
#         • ε (epsilon)   — pequeno offset numérico na raiz quadrada  
#         • γ (gamma)     — peso do “bump” na direção dinâmica ( heading ↔ v )  
#         • ωₘₐₓ          — escala para transição heading → direção v (define α)  
#         • κ (kappa)     — ganho do “puxão” lateral associado a ω  
#         """
#         # --- 1) vetor robô→ponto ---
#         dx = x - self.owner.x
#         dy = y - self.owner.y
#         d2 = dx*dx + dy*dy
#         d  = math.sqrt(d2 + self.epsilon)
#         inv_d = 1.0 / d
#         ex = dx * inv_d
#         ey = dy * inv_d
#         # --- 2) potencial base (gaussiana + velocidade) ---
#         v_x, v_y = self.owner.v_x, self.owner.v_y
#         vdot = v_x*ex + v_y*ey
#         U0 = (
#             self.A
#             * math.exp(-d2 / (2.0 * self.sigma2))
#             * (1.0 + self.beta * vdot)
#         )
#         # --- 3) heading e direção da velocidade ---
#         theta_rad = math.radians(self.owner.theta)
#         hx = math.cos(theta_rad)
#         hy = math.sin(theta_rad)
#         v_lin = math.hypot(v_x, v_y)
#         if v_lin > 1e-6:
#             inv_vlin = 1.0 / v_lin
#             vdir_x = v_x * inv_vlin
#             vdir_y = v_y * inv_vlin
#         else:
#             vdir_x, vdir_y = hx, hy
#         # --- 4) blend heading↔vdir via alpha(omega) ---
#         omega = abs(self.owner.v_theta)
#         alpha = omega / self.omega_max if omega < self.omega_max else 1.0

#         bx = (1.0 - alpha)*hx + alpha*vdir_x
#         by = (1.0 - alpha)*hy + alpha*vdir_y
#         inv_bnorm = 1.0 / math.hypot(bx, by)
#         dxn = bx * inv_bnorm
#         dyn = by * inv_bnorm
#         # --- 5) máscara traseira e dir_factor ---
#         cos_main = dxn*ex + dyn*ey
#         if cos_main <= 0.0:
#             return 0.0
#         dir_factor = 1.0 + self.gamma * cos_main
#         # --- 6) puxão lateral ---
#         px = -hy
#         py = hx
#         turn_sign = (1.0 if self.owner.v_theta > 0
#                      else -1.0 if self.owner.v_theta < 0
#                      else 0.0)
#         cos_turn = (px*ex + py*ey) * turn_sign
#         turn_factor = 1.0 + self.kappa * omega * cos_turn
#         # --- 7) potencial final ---
#         return U0 * dir_factor * turn_factor


#     def gradient_at(self, x: float, y: float) -> tuple[float, float]:
#         """
#         ∇U = [∂U/∂x, ∂U/∂y]
#         derivadas conforme:
#         ∂U/∂x = A * exp(...) * [radial * ex + tang_x], etc.
#         """
#         dx = x - self.owner.x
#         dy = y - self.owner.y
#         d2 = dx*dx + dy*dy
#         d  = np.sqrt(d2 + self.epsilon)
#         ex, ey = dx/d, dy/d

#         # componente de velocidade na direção e
#         vdot = self.owner.v_x * ex + self.owner.v_y * ey
#         exp_term = np.exp(-d2 / (2 * self.sigma**2))

#         # termo radial
#         radial = - (1 + self.beta * vdot) * (d / (self.sigma**2))
#         # termo tangencial (variação com a velocidade perpendicular)
#         tang_x = self.beta * (self.owner.v_x - vdot * ex) / d
#         tang_y = self.beta * (self.owner.v_y - vdot * ey) / d

#         dUdx = self.A * exp_term * (radial * ex + tang_x)
#         dUdy = self.A * exp_term * (radial * ey + tang_y)
#         return dUdx, dUdy

#     def force_at(self, x: float, y: float) -> tuple[float, float]:
#         """
#         F = -∇U
#         """
#         dUdx, dUdy = self.gradient_at(x, y)
#         return -dUdx, -dUdy

#     def influence_on( self, other: 'Robot') -> tuple[float, float]:
#         """
#         Força que este campo (deste robô) exerce em outro robô.
#         """
#         return self.force_at(other.x, other.y)


#     def update_parameters(
#         self,
#         A: float            = None,
#         sigma: float        = None,
#         beta: float         = None,
#         epsilon: float      = None,
#         gamma: float        = None,
#         kappa: float        = None,
#         omega_max: float    = None
#     ):
#         if A         is not None: self.A         = A
#         if beta      is not None: self.beta      = beta
#         if epsilon   is not None: self.epsilon   = epsilon
#         if gamma     is not None: self.gamma     = gamma
#         if kappa     is not None: self.kappa     = kappa
#         if omega_max is not None: self.omega_max = omega_max
#         if sigma     is not None:
#             self.sigma     = sigma
#             self.sigma2    = sigma * sigma
