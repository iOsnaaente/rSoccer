from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.VSSS.vss_gym_base import VSSBaseEnv
from gymnasium.spaces import Box
import numpy as np


class VSSExampleEnv( VSSBaseEnv ):
    def __init__( self, render_mode = None ):

        super().__init__(
            field_type = 0, 
            n_robots_blue = 1,
            n_robots_yellow = 0, 
            time_step = 0.025,
            render_mode = render_mode
        )
        n_obs = 4 # Ball x,y and Robot x, y
        self.action_space = Box( 
            low = -1, 
            high = 1, 
            shape = (2, ) 
        )
        self.observation_space = Box( 
            low = -self.field.length/2,
            high = self.field.length/2,
            shape = ( n_obs, )
        )


    def _frame_to_observations(self):
        ball, robot = self.frame.ball, self.frame.robots_blue[0]
        return np.array([ball.x, ball.y, robot.x, robot.y])


    def _get_commands(self, actions):
        return [Robot(yellow=False, id=0,
                    v_x=actions[0], v_y=actions[1])]

    def _calculate_reward_and_done(self):
        if self.frame.ball.x > self.field.length / 2 \
            and abs(self.frame.ball.y) < self.field.goal_width / 2:
            reward, done = 1, True
        else:
            reward, done = 0, False
        return reward, done
    

    def _get_initial_positions_frame(self):
        pos_frame: Frame = Frame()
        pos_frame.ball = Ball( 
            x = (self.field.length/2) - self.field.penalty_length, 
            y = 0.0 
        )
        pos_frame.robots_blue[0] = Robot( 
            x = 0.0, 
            y = 0.0, 
            theta = 0, 
        )
        return pos_frame


if __name__ == "__main__":
    env = VSSExampleEnv( render_mode = "human" )
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Obs: {obs}, Reward: {reward}, Done: {done}")
    env.close()