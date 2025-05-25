from rSoccer.rsoccer_gym.Entities.Robot import Robot
import numpy as np


class ForceField:
    def __init__( self, owner_robot: Robot, A: float = 1.0, sigma: float = 1.0, beta: float = 1.0, epsilon: float = 1e-6  ):
        self.owner = owner_robot  # instância de Robot a quem este ForceField: pertence                      
        self.epsilon = epsilon    # para evitar divisão por zero                  
        self.sigma = sigma        # raio de influência (desvio padrão do gaussiano)              
        self.beta = beta          # peso do termo de velocidade              
        self.A = A                # amplitude máxima do potencial      

    def potential_at(self, x: float, y: float) -> float:
        """
        U_i(x,y) = A * exp(-d^2/(2σ^2)) * [1 + β * (v · e)],
        onde d = ||r - r_i|| e e = (r - r_i)/d.
        """
        dx = x - self.owner.x
        dy = y - self.owner.y
        d2 = dx*dx + dy*dy
        d = (d2 + self.epsilon)**0.5

        # vetor unitário do obstáculo ao ponto (x,y)
        ex, ey = dx/d, dy/d

        # componente de velocidade na direção e
        vdot = self.owner.vx*ex + self.owner.vy*ey

        return self.A * np.exp(-d2/(2*self.sigma**2)) * (1 + self.beta * vdot)

    def gradient_at(self, x: float, y: float) -> tuple[float,float]:
        """
        ∇U_i = [∂U/∂x, ∂U/∂y] conforme derivadas já discutidas.
        """
        dx = x - self.owner.x
        dy = y - self.owner.y
        d2 = dx*dx + dy*dy
        d = (d2 + self.epsilon)**0.5
        ex, ey = dx/d, dy/d
        vdot = self.owner.vx*ex + self.owner.vy*ey
        exp_term = np.exp(-d2/(2*self.sigma**2))

        # termo base (radial)
        radial = - (1 + self.beta*vdot) * (d / (self.sigma**2))

        # termo de velocidade (tangencial)
        tang_x = self.beta * (self.owner.vx - vdot*ex) / d
        tang_y = self.beta * (self.owner.vy - vdot*ey) / d

        dUdx = self.A * exp_term * ( radial*ex + tang_x )
        dUdy = self.A * exp_term * ( radial*ey + tang_y )
        return dUdx, dUdy

    def force_at(self, x: float, y: float) -> tuple[float,float]:
        """
        F = -∇U
        """
        dUdx, dUdy = self.gradient_at(x, y)
        return -dUdx, -dUdy

    # Métodos adicionais úteis:
    def influence_on(self, other_robot) -> tuple[float,float]:
        """
        Computa a força que este ForceField: (do owner) exerce em other_robot.
        """
        return self.force_at(other_robot.x, other_robot.y)


    def update_parameters(self, *, A=None, sigma=None, beta=None):
        """Permite ajustar A, σ e β em tempo de execução."""
        if A is not None:     self.A = A
        if sigma is not None: self.sigma = sigma
        if beta is not None:  self.beta = beta
