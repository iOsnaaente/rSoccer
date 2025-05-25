from enum import Enum
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.vss.vss_gym_base import VSSBaseEnv
from rsoccer_gym.Utils import KDTree
from rsoccer_gym.vss.env_vss.shared_tcc import (
    observations,
    w_ball_grad_tcc,
    w_energy_tcc,
    w_move_tcc,
    goal_reward,
    own_goal_reward,
)

from rsoccer_gym.Utils.Utils import OrnsteinUhlenbeckAction
from typing import Dict

import numpy as np
import random
import math
import gym


def close_to_x(x, range=0.15):
    return np.clip(x + np.random.uniform(-range, range, 1)[0], -0.5, 0.5)


def close_to_y(x, range=0.15):
    return np.clip(x + np.random.uniform(-range, range, 1)[0], -0.5, 0.5)


def get_smallest_angle(v1, v2):
    angle = math.acos(np.dot(v1, v2))
    if np.cross(v1, v2) > 0:
        return -angle
    return angle


def transform(v1, ang):
    mod = np.sqrt(v1[0] ** 2 + v1[1] ** 2)
    v1 = (v1[0] / mod, v1[1] / mod)

    mn = get_smallest_angle(v1, (math.cos(ang), math.sin(ang)))

    return mn, (math.cos(mn) * mod, math.sin(mn) * mod)


# Enum of difficulties
class Difficulty(Enum):
    INITIAL = 0.1
    EASY = 0.25
    MEDIUM = 0.55
    HARD = 1

    @staticmethod
    def getDifficulty(value):
        if value < Difficulty.EASY.value:
            return Difficulty.INITIAL
        elif value < Difficulty.MEDIUM.value:
            return Difficulty.EASY
        elif value < Difficulty.HARD.value:
            return Difficulty.MEDIUM
        else:
            return Difficulty.HARD


default_metrics = {
    "difficulty": np.array([]),
    "goal": np.array([]),
    "reward": np.array([]),
}


class VSSProgressiveAttackerVSRandomGoalkeeper(VSSBaseEnv):
    """This environment controls a singl
    e robot in a VSS soccer League 3v3 match


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

    def __init__(self):
        super().__init__(
            field_type=0, n_robots_blue=1, n_robots_yellow=3, time_step=0.025
        )

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS, high=self.NORM_BOUNDS, shape=(17,), dtype=np.float32
        )

        self.goleiro_teve_bola = False

        # Initialize Class Atributes
        self.previous_ball_potential = None
        self.actions: Dict = None
        self.reward_shaping_total = None
        self.v_wheel_deadzone = 0.05

        self.ou_actions = []
        for _ in range(self.n_robots_blue + self.n_robots_yellow):
            self.ou_actions.append(
                OrnsteinUhlenbeckAction(self.action_space, dt=self.time_step)
            )

        self.difficulty = 0.0  # starts easy

        self.metrics = default_metrics

        print("Environment initialized")

    def set_diff(self, mean_rewards, max_mean_rewards=450):

        diff = min(
            1, max(0.1, mean_rewards) / max_mean_rewards
        )  # if the mean rewards gets to 450 points, it is maximum difficulty

        print(
            f"Mean rewards: {mean_rewards} | Max mean rewards: {max_mean_rewards} | Diff: {diff}"
        )

        if diff > 0.6 and self.difficulty <= Difficulty.INITIAL.value:
            self.difficulty = Difficulty.EASY.value
        elif (
            diff > 0.7 and Difficulty.getDifficulty(self.difficulty) == Difficulty.EASY
        ):
            self.difficulty = Difficulty.MEDIUM.value
        elif (
            diff > 0.8
            and Difficulty.getDifficulty(self.difficulty) == Difficulty.MEDIUM
        ):
            self.difficulty = Difficulty.HARD.value

    def reset(self):
        self.actions = None
        self.reward_shaping_total = None
        self.previous_ball_potential = None

        for ou in self.ou_actions:
            ou.reset()

        return super().reset()

    def step(self, action):
        observation, reward, done, _ = super().step(action)
        return observation, reward, done, self.reward_shaping_total

    def _frame_to_observations(self):
        return self.attacker_observations()

    def attacker_observations(self):
        return observations(self)

    def _get_goalkeeper_vels(self):
        # Random position to goalkeeper
        actions = self.ou_actions[0].sample()
        v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions)

        return v_wheel0, v_wheel1

    def _get_commands(self, actions):
        commands = []
        self.actions = {}

        self.actions[0] = actions[:2]

        # Attacker actions
        v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions)

        commands.append(Robot(yellow=False, id=0, v_wheel0=v_wheel0, v_wheel1=v_wheel1))

        # Goalkeeper actions
        gk_v_wheel_0, gk_v_wheel_1 = self._get_goalkeeper_vels()

        goalkeeper_move = Robot(
            yellow=True,
            id=0,
            v_wheel0=gk_v_wheel_0 * self.difficulty,
            v_wheel1=gk_v_wheel_1 * self.difficulty,
        )

        commands.append(goalkeeper_move)

        if (
            self.difficulty > 0.5
        ):  # if agent is 50% good, start slowly making other robots move in a random way
            movement = (self.difficulty - 0.2) / 0.8

            # Skip robot with id 0 which is the goalkeeper
            for i in range(1, self.n_robots_yellow):
                actions = self.ou_actions[self.n_robots_blue + i].sample()
                v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions)
                commands.append(
                    Robot(
                        yellow=True,
                        id=i,
                        v_wheel0=v_wheel0 * movement,
                        v_wheel1=v_wheel1 * movement,
                    )
                )

        return commands

    def _calculate_reward_and_done(self):
        reward = 0
        goal = False
        w_move = w_move_tcc
        w_ball_grad = w_ball_grad_tcc
        w_energy = w_energy_tcc
        if self.reward_shaping_total is None:
            self.reward_shaping_total = {
                "move": 0,
                "energy": 0,
                "ball_gradient": 0,
            }

        move_reward = 0
        energy_penalty = 0
        grad_ball_potential = 0

        # Check if goal ocurred
        if self.frame.ball.x > (self.field.length / 2):
            reward = goal_reward
            goal = True
        elif self.frame.ball.x < -(self.field.length / 2):
            reward = own_goal_reward
            goal = True
        else:

            if self.last_frame is not None:
                # Calculate ball potential
                grad_ball_potential = w_ball_grad * self.__ball_grad()
                # Calculate Move ball
                move_reward = w_move * self.__move_reward()
                # Calculate Energy penalty
                energy_penalty = w_energy * self.__energy_penalty()

                reward = grad_ball_potential + move_reward + energy_penalty

                self.reward_shaping_total["move"] += move_reward
                self.reward_shaping_total["energy"] += energy_penalty
                self.reward_shaping_total["ball_gradient"] += grad_ball_potential

        np.append(self.metrics["difficulty"], self.difficulty)
        np.append(self.metrics["reward"], reward)
        np.append(self.metrics["goal"], 1 if goal else 0)

        return reward, goal

    def _get_initial_positions_frame(self):
        """Returns the position of each robot and ball for the initial frame"""

        range_from_point = self.difficulty
        # range_from_point = 1

        def x(x: float = 0):
            return close_to_x(x, range_from_point)

        def y(y: float = 0):
            return close_to_y(y, range_from_point)

        def theta():
            return random.uniform(0, 360)

        pos_frame = Frame()

        places = KDTree()

        pos = (x(), y())
        places.insert(pos)

        pos_frame.ball = Ball(x=pos[0], y=pos[1])

        while places.get_nearest(pos)[1] < 0.1:
            pos = (x(-0.5), y())
        places.insert(pos)

        # Trainer agent initial positions
        pos_frame.robots_blue[0] = Robot(x=pos[0], y=pos[1], theta=theta())

        while places.get_nearest(pos)[1] < 0.1:
            pos = (x(0.6), close_to_y(0, 0.05))
        places.insert(pos)

        # Goalkeeper initial positions - Goal centered
        gk_x = 0.7125  # Parallel to goal line
        gk_y = 0  # Centered in Y axis

        # If difficulty is lower or equal then EASY
        if self.difficulty <= Difficulty.EASY.value:
            # Goalkeeper starts out of the goal - Just their size off
            # Random select to be above or below the goal
            is_above = random.choice([True, False])
            gk_y = (self.field.goal_width / 2) + self.field.rbt_radius
            if is_above:
                gk_y *= -1

        # Goalkeeper starts pointed 90 degrees up
        # If use positive velocities, it goes up in the Y axis
        gk_theta = 90
        pos_frame.robots_yellow[0] = Robot(x=gk_x, y=gk_y, theta=gk_theta)

        while places.get_nearest(pos)[1] < 0.1:
            pos = (x(), y(-0.4))
        places.insert(pos)

        pos_frame.robots_yellow[1] = Robot(x=pos[0], y=pos[1], theta=theta())

        while places.get_nearest(pos)[1] < 0.1:
            pos = (x(), y(0.4))
        places.insert(pos)

        pos_frame.robots_yellow[2] = Robot(x=pos[0], y=pos[1], theta=theta())

        return pos_frame

    def _actions_to_v_wheels(self, actions):
        # Receives values from the network and converts (linear velocity,
        # angular velocity) to wheel speeds between -1 and 1

        # spacing between cart wheels (1 so that the maximum value is 1 too)
        L = 1

        vleft = (actions[0] - (actions[1] * L) / 2) * 1

        vright = (actions[0] + (actions[1] * L) / 2) * 1

        left_wheel_speed = vleft * self.max_v
        right_wheel_speed = vright * self.max_v

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
