from rSoccer.rsoccer_gym.Entities.PotentialField import PotentialField 
from dataclasses import dataclass, field
import numpy as np

@dataclass()
class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Trajectories:
    points: list[Point] = field( default_factory = list )
    goals: list[Point] = field( default_factory = list )

    def __init__(self):
        self.points = []
        self.goals = []

    def add_point(self, point: Point) -> None:
        """Adiciona um ponto à trajetória."""
        self.points.append(point)

    def remove_point(self, index: int) -> None:
        """Remove um ponto pelo índice, se existir."""
        if 0 <= index < len(self.points):
            self.points.pop(index)

    def num_points(self) -> int:
        """Retorna a quantidade de pontos na trajetória."""
        return len(self.points)

    def reset_points(self) -> None:
        """Reseta todos os pontos da trajetória."""
        self.points.clear()

    def add_goal(self, goal: Point) -> None:
        """Adiciona um objetivo à lista de goals."""
        self.goals.append(goal)

    def remove_goal(self, index: int) -> None:
        """Remove um goal pelo índice, se existir."""
        if 0 <= index < len(self.goals):
            self.goals.pop(index)

    def num_goals(self) -> int:
        """Retorna a quantidade de goals."""
        return len(self.goals)

    def reset_goals(self) -> None:
        """Reseta todos os goals."""
        self.goals.clear()


@dataclass
class Robot:
    id: int = None
    yellow: bool = False

    # Posição e orientação no campo
    _x: float = 0.0
    _y: float = 0.0
    
    # Posições auxiliares 
    z: float = 0.0
    theta: float = 0.0

    # Velocidades
    v_x: float = 0.0
    v_y: float = 0.0
    v_theta: float = 0.0

    # Velocidade de chute
    kick_v_x: float = 0.0
    kick_v_z: float = 0.0

    # Ativos/equipamentos
    dribbler: bool = False
    infrared: bool = False
    wheel_speed: bool = False
    
    v_wheel0: float = 0.0  # rad/s
    v_wheel1: float = 0.0  # rad/s
    v_wheel2: float = 0.0  # rad/s
    v_wheel3: float = 0.0  # rad/s

    # Trajetórias e objetivos
    trajectory: Trajectories = field( default_factory = Trajectories )

    # Campo potencial de força relativa a esse Robô 
    force_field: PotentialField = field( init = False )

    def __post_init__(self):
        # Inicializa o Potencial Field associado a este robô
        self.force_field = PotentialField ( owner_robot = self )
    
    @property
    def x(self) -> float:
        return self._x
    
    @x.setter
    def x(self, value: float) -> None:
        self._x = value    
        
    @property
    def y(self) -> float:
        return self._y
    
    @y.setter
    def y(self, value: float) -> None:
        self._y = value

    def update_pose(self, x: float, y: float, z: float, theta: float) -> None:
        """Atualiza a pose do robô."""
        self.x, self.y, self.z, self.theta = x, y, z, theta

    def update_velocity(self, v_x: float, v_y: float, v_theta: float) -> None:
        """Atualiza as velocidades do robô."""
        self.v_x, self.v_y, self.v_theta = v_x, v_y, v_theta

    def record_position(self) -> None:
        """Registra a posição atual na trajetória."""
        if self.trajectory.num_points() > 10:
            self.trajectory.remove_point(0)
        self.trajectory.add_point(Point(self.x, self.y, self.z))

    def record_goal(self, goal: Point) -> None:
        """Registra um novo objetivo atingido."""
        if self.trajectory.num_goals() > 10:
            self.trajectory.remove_goal(0)
        self.trajectory.add_goal( goal )

    def potential_at(self, x: float, y: float) -> float:
        """ Retorna o potencial U no ponto (x,y) calculado pelo campo deste robô. """
        return self.force_field.potential_at(x, y)

    def gradient_at(self, x: float, y: float) -> tuple[float, float]:
        """ Retorna (∂U/∂x, ∂U/∂y) no ponto (x,y). """
        return self.force_field.gradient_at(x, y)

    def force_at(self, x: float, y: float) -> tuple[float, float]:
        """ Retorna a força (Fx, Fy) = -∇U no ponto (x,y) """
        return self.force_field.force_at(x, y)

    def influence_on(self, other: 'Robot') -> tuple[float, float]:
        """ Força que este robô exerce sobre outro na posição dele """
        return self.force_field.influence_on(other)


if __name__ == "__main__":
    '''
    Teste da aplicação do campo potencial para o Robo 
    '''

    # 1. Instancia um robô e define a posição e velocidade dele 
    pos_robot = ( 2.0, 2.0, 0.0 ) # Posição no 1° quadrante 
    vel_robot = ( -1.0, -1.0 )    # Velocidade indo para a origem

    robot = Robot( id = 1, yellow = False )
    robot.update_pose( *pos_robot, theta = 0.0 ) 
    robot.update_velocity( *vel_robot, v_theta = 0.0 )

    # Pontos de teste para calcular o potencial
    '''
          Y
          ▲
          |                      
    3.00  O      O      O      O      
    2.50  |                           
    2.00  O      O      R      O      
    1.50  |                          
    1.00  O      O      O      O       
    0.50  |                           
    0.00  O------O------O------O------►
          0      1      2      3      X
    '''
    test_points = [ (x, y) for y in ( 3.0, 2.0, 1.0, 0.0 ) for x in ( 0.0, 1.0, 2.0, 3.0 ) ]
    print( f"Testando campo de potencial para Robot (ID={robot.id}, yellow={robot.yellow})")

    for px, py in test_points:
        u = robot.potential_at(px, py)
        grad_x, grad_y = robot.gradient_at(px, py)
        force_x, force_y = robot.force_at(px, py)
        print(f"Ponto ({px:.1f}, {py:.1f}):")
        print(f"  Potencial U = {u:.4f}")
        print(f"  Gradiente ∂U/∂x = {grad_x:.4f}, ∂U/∂y = {grad_y:.4f}")
        print(f"  Força Fx = {force_x:.4f}, Fy = {force_y:.4f}\n")