from rSoccer.rsoccer_gym.Entities import Ball, Frame, Robot
import random 

rand_x     = lambda env: random.uniform( -env.field.length / 2 + 0.1, env.field.length / 2 - 0.1 )
rand_y     = lambda env: random.uniform( -env.field.width  / 2 + 0.1, env.field.width  / 2 - 0.1 )
rand_theta = lambda env: random.uniform( 0, 360 )

def init_random( 
    env: 'VSSS_Env', # type: ignore
) -> Frame:
    # Gera numeros aleatórios dentro do Spaw_range 
    # Gera um Frame 
    frame: Frame = Frame()
    # Gera a bola 
    frame.ball = Ball( x = rand_x( env ), y = rand_y( env ) )
    # Robos Azuis 
    for i in range( env.n_robots_blue ):
        frame.robots_blue[i] = Robot( 
            id = i, yellow = False,
            _x = rand_x( env ), _y = rand_y( env ), 
            theta = rand_theta( env ) 
        )
    # Robos amarelos 
    for j in range( env.n_robots_yellow ):
        frame.robots_yellow[j] = Robot( 
            id = j, yellow = True,
            _x = rand_x( env ), _y = rand_y( env ), 
            theta = rand_theta( env ) 
        )
    return frame


def init_corners_1v3(
    env: 'VSSS_Env', # type: ignore
) -> Frame:
    # Gera um Frame para desenhar sobre 
    frame: Frame = Frame() 
    # Para a bola 
    frame.ball = Ball( x = 0.0, y = 0.0 )
    # Para os robôs azuis 
    frame.robots_blue[0] = Robot(
        id = 0, yellow = False, 
        _x = -0.5, _y = -0.5,
        theta = 0, v_theta = 0,
        v_x = 0, v_y = 0,
    )
    # Para os robôs amarelos 
    frame.robots_yellow[0] = Robot(
        id = 0, yellow = True, 
        _x = 0.5, _y = -0.5 , z = 0, 
        theta = 90, v_theta = 0,
        v_x = 0, v_y = -10,
    )
    frame.robots_yellow[1] = Robot(
        id = 1, yellow = True, 
        _x = 0.5, _y = 0.5, z = 0, 
        theta = 180, v_theta = 0,
        v_x = 0, v_y = 10,
    )
    frame.robots_yellow[2] = Robot(
        id = 2, yellow = True, 
        _x = -0.5, _y =  0.5, z = 0, 
        theta = 270, v_theta = 0,
        v_x = 10,  v_y = 0,
    )
    return frame 


def init_align_1v3(
    env: 'VSSS_Env', # type: ignore
) -> Frame:
    # Gera um Frame para desenhar sobre 
    frame: Frame = Frame() 
    # Para a bola 
    frame.ball = Ball( x = 0.50, y = 0.0 )
    # Para os robôs azuis 
    frame.robots_blue[0] = Robot(
        id = 0, yellow = False, 
        _x = -0.5, _y = 0.0,
        theta = 0, v_theta = 0,
        v_x = 0, v_y = 0,
    )
    # Para os robôs amarelos 
    frame.robots_yellow[0] = Robot(
        id = 0, yellow = True, 
        _x = -0.25, _y = 0.0, 
        theta = 0.0, v_theta = 0
    )
    frame.robots_yellow[1] = Robot(
        id = 1, yellow = True, 
        _x = 0.0, _y = 0.0, 
        theta = 0.0, v_theta = 0
    )
    frame.robots_yellow[2] = Robot(
        id = 2, yellow = True, 
        _x = 0.25, _y =  0.0, 
        theta = 0.0, v_theta = 0
    )
    return frame 



def init_align_Y_1v3(
    env: 'VSSS_Env', # type: ignore
) -> Frame:
    # Gera um Frame para desenhar sobre 
    frame: Frame = Frame() 
    # Para a bola 
    frame.ball = Ball( x = 0.50, y = 0.0 )
    # Para os robôs azuis 
    frame.robots_blue[0] = Robot(
        id = 0, yellow = False, 
        _x = -0.5, _y = 0.0,
        theta = 0, v_theta = 0,
        v_x = 0, v_y = 0,
    )
    # Para os robôs amarelos 
    frame.robots_yellow[0] = Robot(
        id = 0, yellow = True, 
        _x = -0.25, _y = 0.0, 
        theta = 0.0, v_theta = 0
    )
    frame.robots_yellow[1] = Robot(
        id = 1, yellow = True, 
        _x = 0.0, _y = 0.30, 
        theta = 0.0, v_theta = 0
    )
    frame.robots_yellow[2] = Robot(
        id = 2, yellow = True, 
        _x = 0.25, _y =  0.0, 
        theta = 0.0, v_theta = 0
    )
    return frame 
