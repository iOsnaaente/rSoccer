from numba import njit
import math


@njit( cache = False )
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
                    omega_max: float,
                    k_stretch: float,
                    v_lin_max: float ) -> float:

        """
        Potencial U(x,y) = U₀(x,y) · F_dir(x,y) · F_turn(x,y)
        Onde:
        
        U₀(x,y) = A · exp(-d² / (2·σ²)) · [1 + β · (v · e)]
            • d   = distância do robô ao ponto = √(dx² + dy² + ε)
            • e   = vetor unitário robo→ponto
            • v·e = projeção da velocidade linear no radial

        F_dir(x,y) = 1 + γ · (d̂ · e)
            • d̂ = vetor de responsividade “dinâmico”, mistura entre
                - h = (cos θ, sin θ) (heading do robô) e
                - v̂ = direção da velocidade linear
            por um peso α = min(|ω|/ωₘₐₓ, 1)
            • γ controla quão forte o campo “puxa” para essa direção

        F_turn(x,y) = 1 + κ · |ω| · [sign(ω)·(p · e)]
            • p = vetor perpendicular a h (lado esquerdo)
            • sign(ω) escolhe esquerda (+) ou direita (-)
            • κ controla a intensidade do “puxão” lateral
            
        Se (d̂·e) < 0, retornamos 0 para eliminar todo campo atrás do robô.

        **Variáveis de tuning disponíveis:**
        • A             — amplitude do potencial gaussiano  
        • σ (sigma)     — largura do “bump” gaussiano  
        • β (beta)      — ganho do termo de velocidade linear  
        • ε (epsilon)   — pequeno offset numérico na raiz quadrada  
        • γ (gamma)     — peso do “bump” na direção dinâmica ( heading ↔ v )  
        • ωₘₐₓ          — escala para transição heading → direção v (define α)  
        • κ (kappa)     — ganho do “puxão” lateral associado a ω  
        """

        # Normalização radial
        d2 = dx*dx + dy*dy
        d  = math.sqrt( d2 + epsilon)

        # Vetor unitário do robô para o ponto (x,y)
        ex, ey = dx/d, dy/d

        # Componente de velocidade na direção e
        vdot = v_x * ex + v_y * ey
        
        # Calculo do campo potencial 
        U0 = A * math.exp(-d2 / (2.0 * sigma2)) * (1.0 + beta * vdot)

        # Cálculo dos vetores unitários de heading e direção de velocidade
        #    - h = (hx, hy) representa a orientação atual do robô (heading)
        #    - vdir = (vdir_x, vdir_y) representa a direção do vetor de velocidade linear
        #    Se v_lin > 1e-6, normaliza (v_x, v_y); caso contrário, evita divisão por zero
        #    e usa h como fallback.
        theta_rad = math.radians(theta_deg)
        hx = math.cos(theta_rad)
        hy = math.sin(theta_rad)
        
        v_lin = math.hypot(v_x, v_y)

        if v_lin > 1e-3:
            inv_vlin = 1.0 / v_lin
            vdir_x = v_x * inv_vlin
            vdir_y = v_y * inv_vlin
        else:
            vdir_x, vdir_y = hx, hy

        # Escalar efeitos por velocidade linear
        lin_scale = v_lin / v_lin_max if v_lin < v_lin_max else 1.0

        # Blend heading ↔ vdir via alpha(omega)
        omega = abs(v_theta)
        alpha = omega / omega_max if omega < omega_max else 1.0
        bx = (1.0 - alpha)*hx + alpha*vdir_x
        by = (1.0 - alpha)*hy + alpha*vdir_y
        inv_bnorm = 1.0 / math.hypot(bx, by)
        dxn = bx * inv_bnorm
        dyn = by * inv_bnorm

        # Máscara traseira e fator de orientação
        cos_main = dxn*ex + dyn*ey
        if cos_main <= 0.0:
            dir_factor = 1.0
        else: 
            dir_factor = 1.0 + gamma * cos_main * lin_scale

        # Puxão lateral
        px = -hy
        py = hx
        turn_sign = 1.0 if v_theta > 0.0 else -1.0 if v_theta < 0.0 else 0.0
        cos_turn = (px*ex + py*ey) * turn_sign
        turn_factor = 1.0 + kappa * omega * cos_turn * lin_scale
        U_ori = dir_factor * turn_factor 

        # 4) projeções
        proj_par = dx*dxn + dy*dyn
        proj_per = dx*px  + dy*py

        # 5) desvios: estica ao longo de d̂ com v_lin
        sigma = math.sqrt(sigma2)
        sigma_par = sigma + k_stretch * min(v_lin, v_lin_max)
        sigma_per = 0.5*sigma

        # 6) half–Gaussian elipsoidal
        E_hg = math.exp( -0.5 * ( (proj_par/sigma_par)**2 + (proj_per/sigma_per)**2 ))
        if E_hg < 0.0:
            E_hg = 0.0 
        
        # 7) Retorna o potencial total
        return U0 * ( 1 + U_ori ) * ( 1 + E_hg ) 


@njit(cache=True)
def _gradient_core(dx: float,
                   dy: float,
                   v_x: float,
                   v_y: float,
                   A: float,
                   sigma: float,
                   beta: float,
                   epsilon: float) -> tuple:
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

    dUdx: float = A * exp_term * (radial * ex + tang_x)
    dUdy: float = A * exp_term * (radial * ey + tang_y)
    return ( dUdx, dUdy )


class PotentialField:
    def __init__(
        self, 
        owner_robot,

        A: float = 1.0,
        sigma: float = 0.1,
        beta: float = 5.0,
        epsilon: float = 1e-5,
        gamma: float = 0.001,
        kappa: float = 0.0125,
        omega_max: float = 30.0,
        influence_scale: float = 3.0,
        v_lin_max: float = 1.0,
        k_stretch: float = 10,
):

        self.owner = owner_robot
        self.A = A
        self.sigma = sigma
        self.beta = beta
        self.epsilon = epsilon
        self.gamma = gamma
        self.kappa = kappa
        self.omega_max = omega_max
        self.sigma2 = sigma * sigma
        self.v_lin_max = v_lin_max
        self.k_stretch = k_stretch

        # Define raio de influência (padrão 3*sigma)
        self.influence_radius2 = ( influence_scale * self.sigma)**2

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
            self.omega_max,
            self.k_stretch,
            self.v_lin_max
        )

    def gradient_at( self, x: float, y: float) -> tuple[float, float]:
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
        omega_max: float = None,
        k_stretch: float = None,
        v_lin_max: float = None  
    ) -> None:
        if A is not None:           self.A = A
        if beta is not None:        self.beta = beta
        if epsilon is not None:     self.epsilon = epsilon
        if gamma is not None:       self.gamma = gamma
        if kappa is not None:       self.kappa = kappa
        if omega_max is not None:   self.omega_max = omega_max
        if k_stretch is not None:   self.k_stretch = k_stretch
        if v_lin_max is not None:   self.v_lin_max = v_lin_max
        if sigma is not None:
            self.sigma = sigma
            self.sigma2 = sigma * sigma