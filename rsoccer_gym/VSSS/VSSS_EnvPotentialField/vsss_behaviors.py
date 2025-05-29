from rSoccer.rsoccer_gym.Entities import Ball, Frame, Robot
import random 

def go_to_ball_all_1v3( env, commands ) -> None:
    # ROBÔS AMARELOS 
    for ry in range( env.n_robots_yellow ):
        robot = env.frame.robots_yellow[ry]
        v_r, v_l = env.go_to_ball( robot )
        v_wheel0, v_wheel1 = env._actions_to_v_wheels( (v_r, v_l) )
        commands.append(
            Robot( 
                yellow = True, 
                id = ry, 
                v_wheel0 = v_wheel0, 
                v_wheel1 = v_wheel1
            )
        )
    # ROBÔS AZUIS
    if env.n_robots_blue > 1:
        for rb in range(1, env.n_robots_blue):
            robot = env.frame.robots_blue[rb]
            v_r, v_l = env.go_to_ball( robot )
            v_wheel0, v_wheel1 = env._actions_to_v_wheels( (v_r, v_l) )
            commands.append(
                Robot( 
                    yellow = False, 
                    id = rb, 
                    v_wheel0 = v_wheel0, 
                    v_wheel1 = v_wheel1
                )
            )


def stopped_all( env, commands ) -> None:
    # ROBÔS AMARELOS 
    for ry in range( env.n_robots_yellow ):
        commands.append(
            Robot( 
                yellow = True, 
                id = ry, 
                v_wheel0 = 0.0, 
                v_wheel1 = 0.0
            )
        )
    # ROBÔS AZUIS
    if env.n_robots_blue > 1:
        for rb in range(1, env.n_robots_blue):
            commands.append(
                Robot( 
                    yellow = False, 
                    id = rb, 
                    v_wheel0 = 0.0, 
                    v_wheel1 = 0.0
                )
            )