from rSoccer.rsoccer_gym.Entities import Ball, Frame, Robot
from rSoccer.rsoccer_gym.VSSS.env_vss.vss_gym import VSSEnv

from gymnasium.spaces import Box
import numpy as np

class VSSS_Env( VSSEnv ):

    def __init__( self, field_type, n_robots_blue, n_robots_yellow, time_step = 0.025, render_mode = None ):
        self.field_type = field_type 
        self.n_robots_blue = n_robots_blue 
        self.n_robots_yellow = n_robots_yellow 
        self.time_step = time_step 
        self.render_mode = render_mode

        super().__init__(
            field_type = self.field_type, 
            n_robots_blue = self.n_robots_blue,
            n_robots_yellow = self.n_robots_yellow, 
            time_step = self.time_step,
            render_mode = self.render_mode
        )

        # Ball x, y and Robot x, y
        n_obs = 4 
        self.action_space = Box( low = -1, high = 1, shape = (2, ) )
        self.observation_space = Box( 
            low = -self.field.length / 2,
            high = self.field.length / 2,
            shape = ( n_obs, )
        )


    def _frame_to_observations(self):
        ball, robot = self.frame.ball, self.frame.robots_blue[0]
        return np.array([ball.x, ball.y, robot.x, robot.y])


    def _get_commands( self, actions ):
        return [ 
            Robot( 
                yellow = False, 
                id = 0,
                v_x = actions[0], 
                v_y = actions[1]
            ) 
        ]


    def _calculate_reward_and_done(self):
        if self.frame.ball.x > ( self.field.length / 2 ) and abs( self.frame.ball.y) < (self.field.goal_width / 2):
            reward, done = 1, True
        else:
            reward, done = 0, False
        return reward, done
    

    def _get_initial_positions_frame(self):
        pos_frame = Frame()

        # Bola no meio do campo, recuado pela distância de pênalti
        pos_frame.ball = Ball( 
            x = (self.field.length / 2) - self.field.penalty_length, 
            y = 0.0 
        )

        # Distribuição dos robôs azuis
        n_blue   = self.n_robots_blue
        spacing_blue = self.field.width / (n_blue + 1)
        x_blue   = -self.field.length / 4  # exemplo: na vertical esquerda
        for i in range(n_blue):
            y = -self.field.width / 2 + spacing_blue * (i + 1)
            pos_frame.robots_blue[i] = Robot(
                id = i,
                yellow = False,
                _x = x_blue,
                _y = y,
                theta = 0.0
            )

        # Distribuição dos robôs amarelos
        n_yellow   = self.n_robots_yellow
        spacing_yellow = self.field.width / (n_yellow + 1)
        x_yellow   = self.field.length / 4  # exemplo: na vertical direita
        for i in range(n_yellow):
            y = -self.field.width / 2 + spacing_yellow * (i + 1)
            pos_frame.robots_yellow[i] = Robot(
                id = i,
                yellow = True,
                _x = x_yellow,
                _y = y,
                theta = np.pi  # de frente para o outro lado
            )
        return pos_frame