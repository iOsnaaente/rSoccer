from rSoccer.rsoccer_gym.Utils.Utils import OrnsteinUhlenbeckAction
from rSoccer.rsoccer_gym.Entities import Ball, Frame, Robot
from rSoccer.rsoccer_gym.VSSS.VSSS_base import VSSBaseEnv
from rSoccer.rsoccer_gym.Utils import KDTree

import gymnasium as gym
import numpy as np
import random
import math


MATCH_3V3 = 0 
MATCH_5V5 = 1


def _normalize_angle(angle: float) -> float:
    """
    Normaliza ângulo para o intervalo [-π, π]
    """
    return (angle + math.pi) % (2*math.pi) - math.pi


"""
    This environment controls a single robot in a VSS soccer League 3v3 match
    The robot is controlled by two wheel speeds, and the goal is to move the robot
    towards the ball.
    The environment is based on the VSSS soccer league, which is a simplified version
    to test the Gradient Potential Algorithm.

    Description:

    Observation:
        Type: Box(40)
        Normalized Bounds to [-1.25, 1.25]
        Num             Observation normalized
        0               Ball X
        1               Ball Y
        2               Ball Vx
        3               Ball Vy
        4 + (7 * i)     id i Blue Robot X
        5 + (7 * i)     id i Blue Robot Y
        6 + (7 * i)     id i Blue Robot sin(theta)
        7 + (7 * i)     id i Blue Robot cos(theta)
        8 + (7 * i)     id i Blue Robot Vx
        9  + (7 * i)    id i Blue Robot Vy
        10 + (7 * i)    id i Blue Robot v_theta
        25 + (5 * i)    id i Yellow Robot X
        26 + (5 * i)    id i Yellow Robot Y
        27 + (5 * i)    id i Yellow Robot Vx
        28 + (5 * i)    id i Yellow Robot Vy
        29 + (5 * i)    id i Yellow Robot v_theta

    Actions:
        Type: Box(2, )
        Num     Action
        0       id 0 Blue Left Wheel Speed  (%)
        1       id 0 Blue Right Wheel Speed (%)

    Reward:
        Sum of Rewards:
            Goal
            Ball Potential Gradient
            Move to Ball
            Energy Penalty

    Starting State:
        Randomized Robots and Ball initial Position
    
    Episode Termination:
        5 minutes match time
"""

class VSSS_Env( VSSBaseEnv ):

    def __init__( self, 
        field_type: int, 
        n_robots_blue: int, 
        n_robots_yellow: int, 
        time_step: float = 0.025, 
        render_mode: str | None = None 
    ):
        # Condiciona os dados de entrada 
        self.render_mode = "rgb_array" if render_mode == None else "human"
        self.field_type = 0 if field_type  < 0 else 1 if field_type > 1 else field_type 
        self.n_robots_blue = 1 if n_robots_blue < 1 else 3 if n_robots_blue > 3 else n_robots_blue 
        self.n_robots_yellow = 0 if n_robots_yellow < 0 else 3 if n_robots_yellow > 3 else n_robots_yellow
        self.time_step = time_step 

        print( f"Field Type: {self.field_type}")
        print( f"Blue Robots: {self.n_robots_blue}, Yellow Robots: {self.n_robots_yellow}" )
        print( f"Time Step: {self.time_step}, Render Mode: {self.render_mode}" )

        super().__init__(
            field_type = self.field_type, 
            n_robots_blue = self.n_robots_blue,
            n_robots_yellow = self.n_robots_yellow, 
            time_step = self.time_step,
            render_mode = self.render_mode
        )

        # Actions para controle do Robo Azul de ID = 0
        self.action_space = gym.spaces.Box( low = -1, high = 1, shape = (2,), dtype = np.float32 )
        
        # Observações do ambiente descritos na documentação da classe 
        self.observation_space = gym.spaces.Box(
            low = -self.NORM_BOUNDS, 
            high = self.NORM_BOUNDS, 
            shape = (40,), 
            dtype = np.float32
        )

        # Initialize Class Atributes
        self.previous_ball_potential: float = None
        self.reward_shaping_total: float = None
        self.v_wheel_deadzone: float = 0.05
        self.actions: dict = None

        self.ou_actions = []
        for i in range( self.n_robots_blue + self.n_robots_yellow ):
            self.ou_actions.append(
                OrnsteinUhlenbeckAction( self.action_space, dt = self.time_step )
            )


    def reset( self, *, seed = None, options = None ):
        self.previous_ball_potential = None
        self.reward_shaping_total = None
        self.actions = None
        for ou in self.ou_actions:
            ou.reset()
        return super().reset( seed = seed, options = options )


    def step( self, action ):
        observation, reward, terminated, truncated, _ = super().step(action)
        return observation, reward, terminated, truncated, self.reward_shaping_total


    # Observação da posição dos robôs e bola  
    def _frame_to_observations(self):
        observation = []
        observation.append(self.norm_pos(self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        observation.append(self.norm_v(self.frame.ball.v_x))
        observation.append(self.norm_v(self.frame.ball.v_y))
        for i in range(self.n_robots_blue):
            observation.append(self.norm_pos(self.frame.robots_blue[i].x))
            observation.append(self.norm_pos(self.frame.robots_blue[i].y))
            observation.append(np.sin(np.deg2rad(self.frame.robots_blue[i].theta)))
            observation.append(np.cos(np.deg2rad(self.frame.robots_blue[i].theta)))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_x))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_y))
            observation.append(self.norm_w(self.frame.robots_blue[i].v_theta))
        for i in range(self.n_robots_yellow):
            observation.append(self.norm_pos(self.frame.robots_yellow[i].x))
            observation.append(self.norm_pos(self.frame.robots_yellow[i].y))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_x))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_y))
            observation.append(self.norm_w(self.frame.robots_yellow[i].v_theta))
        return np.array( observation, dtype = np.float32 )
    

    def _get_commands( self, actions ):
        self.actions: dict = {}
        commands: list = []
    
        # Ação do Robo Azul de ID = 0
        self.actions[0] = actions
        v_wheel0, v_wheel1 = self._actions_to_v_wheels( actions )
        commands.append( 
            Robot(
                yellow = False, 
                id = 0, 
                v_wheel0 = v_wheel0, 
                v_wheel1 = v_wheel1
            )
        )

        # PARA MUDAR O COMPORTAMENTO DOS ROBÔS AMARELOS 
        robot = self.frame.robots_yellow[0]
        commands.append(
            Robot( 
                yellow = True, 
                id = 0, 
                v_wheel0 = 0, 
                v_wheel1 = 0
            )
        )

        robot = self.frame.robots_yellow[1]
        commands.append(
            Robot( 
                yellow = True, 
                id = 1, 
                v_wheel0 = 45, 
                v_wheel1 = 45
            )
        )

        robot = self.frame.robots_yellow[2]
        v_r, v_l = self.go_to_ball( robot )
        commands.append(
            Robot( 
                yellow = True, 
                id = 2, 
                v_wheel0 = v_r, 
                v_wheel1 = v_l*50
            )
        )

        # PARA MUDAR O COMPORTAMENTO DOS ROBÔS AZUIS
        for i in range(1, self.n_robots_blue):
            actions = self.ou_actions[i].sample()
            self.actions[i] = actions
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions)
            commands.append(
                Robot(
                    yellow = False, 
                    id = i, 
                    v_wheel0 = v_wheel0, 
                    v_wheel1 = v_wheel1
                )
            )
        return commands


    # Calcula a recompensa e se o jogo terminou
    def _calculate_reward_and_done(self):
        w_ball_grad = 0.8
        w_energy = 2e-4
        w_move = 0.2
        reward = 0
        goal = False

        if self.reward_shaping_total is None:
            self.reward_shaping_total = {
                "goal_score": 0,
                "move": 0,
                "ball_grad": 0,
                "energy": 0,
                "goals_blue": 0,
                "goals_yellow": 0,
            }

        # Check if goal ocurred
        if self.frame.ball.x > (self.field.length / 2):
            self.reward_shaping_total["goal_score"] += 1
            self.reward_shaping_total["goals_blue"] += 1
            reward = 10
            goal = True
        elif self.frame.ball.x < -(self.field.length / 2):
            self.reward_shaping_total["goal_score"] -= 1
            self.reward_shaping_total["goals_yellow"] += 1
            reward = -10
            goal = True
        else:
            if self.last_frame is not None:
                # Calculate ball potential
                grad_ball_potential = self.__ball_grad()
                # Calculate Move ball
                move_reward = self.__move_reward()
                # Calculate Energy penalty
                energy_penalty = self.__energy_penalty()

                reward = (
                    w_move * move_reward
                    + w_ball_grad * grad_ball_potential
                    + w_energy * energy_penalty
                )

                self.reward_shaping_total["move"] += w_move * move_reward
                self.reward_shaping_total["ball_grad"] += (
                    w_ball_grad * grad_ball_potential
                )
                self.reward_shaping_total["energy"] += w_energy * energy_penalty

        return reward, goal
    

    def _get_initial_positions_frame(self):
        """Returns the position of each robot and ball for the initial frame"""
        # x     = lambda: random.uniform( -self.field.length / 2 + 0.1, self.field.length / 2 - 0.1 )
        # y     = lambda: random.uniform( -self.field.width  / 2 + 0.1, self.field.width  / 2 - 0.1 )
        # theta = lambda: random.uniform( 0, 360 )
        pos_frame: Frame = Frame()
        places = KDTree()
        
        # Para a bola 
        pos_frame.ball = Ball( 
            x = 0.5, y = 0.0 
        )
        places.insert( (pos_frame.ball.x, pos_frame.ball.y) )

        # Para os robôs azuis 
        pos_frame.robots_blue[0] = Robot(
            id = 0, 
            yellow = False, 
            _x = -0.5,
            _y = 0, 
            z = 0, 
            theta = 0,
            v_x = 0, 
            v_y = 0,
            v_theta = 0
        )
        # for i in range( self.n_robots_blue ):
        #     x_pos, y_pos = x(), y()
        #     while places.get_nearest( (x_pos, y_pos) )[1] < min_dist:
        #         x_pos, y_pos = x(), y()
        #     pos_frame.robots_blue[i] = Robot( 
        #         _x = x_pos, 
        #         _y = y_pos, 
        #         theta = theta() 
        #     )


        # # Para os robôs amarelos 
        pos_frame.robots_yellow[0] = Robot(
            id = 0, 
            yellow = True, 
            _x = 0.0,
            _y = 0.5, 
            z = 0, 
            theta = 270,
            v_x = 0, 
            v_y = -10,
            v_theta = 0
        )
        pos_frame.robots_yellow[1] = Robot(
            id = 1, 
            yellow = True, 
            _x = 0.0,
            _y = -0.50, 
            z = 0, 
            theta = 90,
            v_x = 0, 
            v_y = 10,
            v_theta = 0
        )
        pos_frame.robots_yellow[2] = Robot(
            id = 2, 
            yellow = True, 
            _x = -0.5,
            _y = -0.5, 
            z = 0, 
            theta = 0,
            v_x = 10, 
            v_y = 0,
            v_theta = 0
        )
        # for j in range( self.n_robots_yellow ):
        #     x_pos, y_pos = x(), y()
        #     while places.get_nearest( (x_pos, y_pos) )[1] < min_dist:
        #         x_pos, y_pos = x(), y()
        #     pos_frame.robots_yellow[j] = Robot( 
        #         _x = x_pos, 
        #         _y = y_pos, 
        #         theta = theta() 
        #     )
        return pos_frame


    def _actions_to_v_wheels(self, actions):
        left_wheel_speed = actions[0] * self.max_v
        right_wheel_speed = actions[1] * self.max_v
        left_wheel_speed, right_wheel_speed = np.clip(
            (left_wheel_speed, right_wheel_speed), -self.max_v, self.max_v
        )
        # Deadzone
        if -self.v_wheel_deadzone < left_wheel_speed < self.v_wheel_deadzone:
            left_wheel_speed = 0
        if -self.v_wheel_deadzone < right_wheel_speed < self.v_wheel_deadzone:
            right_wheel_speed = 0
        # Convert to rad/s
        left_wheel_speed /= self.field.rbt_wheel_radius
        right_wheel_speed /= self.field.rbt_wheel_radius
        return left_wheel_speed, right_wheel_speed


    def __ball_grad(self):
        """Calculate ball potential gradient
        Difference of potential of the ball in time_step seconds.
        """
        # Calculate ball potential
        length_cm = self.field.length * 100
        half_lenght = (self.field.length / 2.0) + self.field.goal_depth
        # distance to defence
        dx_d = (half_lenght + self.frame.ball.x) * 100
        # distance to attack
        dx_a = (half_lenght - self.frame.ball.x) * 100
        dy = (self.frame.ball.y) * 100
        dist_1 = -math.sqrt(dx_a**2 + 2 * dy**2)
        dist_2 = math.sqrt(dx_d**2 + 2 * dy**2)
        ball_potential = ((dist_1 + dist_2) / length_cm - 1) / 2
        grad_ball_potential = 0
        # Calculate ball potential gradient
        # = actual_potential - previous_potential
        if self.previous_ball_potential is not None:
            diff = ball_potential - self.previous_ball_potential
            grad_ball_potential = np.clip(diff * 3 / self.time_step, -5.0, 5.0)
        self.previous_ball_potential = ball_potential
        return grad_ball_potential


    def __move_reward(self):
        """Calculate Move to ball reward
        Cosine between the robot vel vector and the vector robot -> ball.
        This indicates rather the robot is moving towards the ball or not.
        """

        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        robot = np.array([self.frame.robots_blue[0].x, self.frame.robots_blue[0].y])
        robot_vel = np.array(
            [self.frame.robots_blue[0].v_x, self.frame.robots_blue[0].v_y]
        )
        robot_ball = ball - robot
        robot_ball = robot_ball / np.linalg.norm(robot_ball)
        move_reward = np.dot(robot_ball, robot_vel)
        move_reward = np.clip(move_reward / 0.4, -5.0, 5.0)
        return move_reward


    def __energy_penalty(self):
        """Calculates the energy penalty"""
        en_penalty_1 = abs(self.sent_commands[0].v_wheel0)
        en_penalty_2 = abs(self.sent_commands[0].v_wheel1)
        energy_penalty = -(en_penalty_1 + en_penalty_2)
        return energy_penalty

    
    def go_to_ball(
        self,
        robot: Robot,
        K_lin: float = 1.0,
        K_ang: float = 4.0
    ) -> np.ndarray:
        """
        Retorna um vetor [a_l, a_r] em [-1,1] prontos para _actions_to_v_wheels,
        que faz o robô aproximar-se da bola com controle proporcional em distância
        e ângulo.
        """
        # 1) Posição bola e robô
        ball = self.frame.ball
        dx = ball.x - robot.x
        dy = ball.y - robot.y
        dist = math.hypot(dx, dy)

        # 2) Se já está quase em cima, para
        if dist < 1e-2:
            return np.array([0.0, 0.0], dtype=float)

        # 3) Ângulo desejado (radianos) e erro de heading
        angle_to_ball = math.atan2(dy, dx)                # em rad
        theta_rad = math.radians(robot.theta)             # converte graus → rad
        err = _normalize_angle(angle_to_ball - theta_rad)

        # 4) Lei de controle P
        v = K_lin * dist             # velocidade linear desejada (m/s)
        omega = K_ang * err          # velocidade angular desejada (rad/s)

        # 5) Converte em velocidades de roda (m/s)
        L = self.field.rbt_wheel_radius
        v_r = v + omega * (L/2)
        v_l = v - omega * (L/2)

        # 6) Normaliza para [-max_v, max_v]
        max_v = self.max_v
        m = max(abs(v_l), abs(v_r), max_v)
        v_l = v_l * max_v / m
        v_r = v_r * max_v / m

        # 7) Retorna normalizado em [-1,1] para cada roda
        return np.array([v_l / max_v, v_r / max_v], dtype=float)