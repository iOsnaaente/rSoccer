from rSoccer.rsoccer_gym.PathPlanning.WeightedRRT import WeightedRRTPlanner 

from rSoccer.rsoccer_gym.Simulators.rsim import RSimVSS

from rSoccer.rsoccer_gym.Entities import PotentialField
from rSoccer.rsoccer_gym.Entities import Frame
from rSoccer.rsoccer_gym.Entities import Robot
from rSoccer.rsoccer_gym.Entities import Field

from rSoccer.rsoccer_gym.Render import VSSRenderField 
from rSoccer.rsoccer_gym.Render import VSSRobot
from rSoccer.rsoccer_gym.Render import COLORS 
from rSoccer.rsoccer_gym.Render import Ball

from rSoccer.rsoccer_gym.VSSS.VSSS_accel import _compute_grid_centers_core
from rSoccer.rsoccer_gym.VSSS.VSSS_accel import _compute_heatmap_core 

import gymnasium as gym
import numpy as np
import pygame


class VSSBaseEnv( gym.Env ):
    metadata = {
        "render.modes": [ "human", "rgb_array" ],
        "render_modes": [ "human", "rgb_array" ],
        "render_fps"  : 60,
        "render.fps"  : 60,
    }
    NORM_BOUNDS = 1.2

    def __init__(
        self,
        field_type: int,
        n_robots_blue: int,
        n_robots_yellow: int,
        time_step: float,
        render_mode = None,
        grid_spacing: float = 0.01,

    ):
        
        # Initialize Simulator
        super().__init__()
        self.n_robots_yellow = n_robots_yellow
        self.n_robots_blue = n_robots_blue
        self.render_mode = render_mode
        self.field_type = field_type
        self.time_step = time_step

        # Inicializa o rSIM - Simulator de VSSS
        self.rsim = RSimVSS(
            field_type = field_type,
            n_robots_blue = n_robots_blue,
            n_robots_yellow = n_robots_yellow,
            time_step_ms = int( self.time_step * 1000 ),
        )

        # Get field dimensions
        self.field = self.rsim.get_field_params()
        self.max_pos = max(
            self.field.width / 2, (self.field.length / 2) + self.field.penalty_length
        )
        max_wheel_rad_s = (self.field.rbt_motor_max_rpm / 60) * 2 * np.pi
        self.max_v = max_wheel_rad_s * self.field.rbt_wheel_radius

        # 0.04 = robot radius (0.0375) + wheel thicknees (0.0025)
        self.max_w = np.rad2deg(self.max_v / 0.04)

        # Initiate
        self.last_frame: Frame = None
        self.frame: Frame = None
        self.sent_commands = None
        self.steps = 0

        # Render
        self.field_renderer = VSSRenderField()
        self.static_field_surface = self.field_renderer.static_surface()
        self.window_surface = None
        self.window_size = self.field_renderer.window_size
        self.field_size = self.field_renderer.field_size
        self.window_pos = self.field_renderer.window_pos
        self.window_id = self.field_renderer.window_id
        self.clock = None

        # calcula grid centers com Numba
        half_length = self.field.length/2
        half_width  = self.field.width/2
        self.grid_spacing = grid_spacing
        self.centers = _compute_grid_centers_core( half_length, half_width, self.grid_spacing )

        # Inicializa o pathPlanning 
        self.path_planning_method: WeightedRRTPlanner = None 
        self.path: list = []



    def _get_commands(self, action):
        """returns a list of commands of type List[Robot] from type action_space action"""
        raise NotImplementedError

    def _frame_to_observations(self):
        """returns a type observation_space observation from a type List[Robot] state"""
        raise NotImplementedError

    def _calculate_reward_and_done(self):
        """returns reward value and done flag from type List[Robot] state"""
        raise NotImplementedError

    def _get_initial_positions_frame(self) -> Frame:
        """returns frame with robots initial positions"""
        raise NotImplementedError

    def step( self, action ):
        self.steps += 1

        # Join agent action with environment actions
        commands: list[Robot] = self._get_commands(action)
        # Send command to simulator
        self.rsim.send_commands(commands)
        self.sent_commands = commands
        # Get Frame from simulator
        self.last_frame = self.frame
        self.frame = self.rsim.get_frame()
        # Calculate environment observation
        observation = self._frame_to_observations()
        # Calculate environment reward and done condition
        reward, done = self._calculate_reward_and_done()
        if self.render_mode == "human":
            self.render()
        return observation, reward, done, False, {}

    def reset(self, *, seed = None, options = None ):
        super().reset(seed = seed, options = options )
        self.sent_commands = None
        self.last_frame = None
        self.steps = 0

        initial_pos_frame: Frame = self._get_initial_positions_frame()
        self.rsim.reset(initial_pos_frame)

        # Get frame from simulator
        self.frame = self.rsim.get_frame()

        obs = self._frame_to_observations()
        if self.render_mode == "human":
            self.render()
        return obs, {}


    def pos_transform( self, pos_x, pos_y ):
        return (
            int(pos_x * self.field_renderer.scale + self.field_renderer.center_x),
            int(pos_y * self.field_renderer.scale + self.field_renderer.center_y),
        )

    def render(self) -> None:
        """
        Renders the game depending on
        ball's and players' positions.
        """

        if self.window_surface is None:
            # Para tirar do modo Full Screen é só retirar aqui 
            import os 
            os.environ['SDL_VIDEO_FULLSCREEN_DISPLAY'] = str( self.window_id ) 
            pygame.init()

            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption( "VSSS Environment" )
                self.window_surface = pygame.display.set_mode( 
                    self.window_size, 
                    flags = pygame.FULLSCREEN | pygame.SCALED,
                    depth = 32, # 32 para canal Alpha ( R G B A )
                    display = self.window_id
                )
            elif self.render_mode == "rgb_array":
                self.window_surface = pygame.Surface( self.window_size )
        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."
        if self.clock is None:
            self.clock = pygame.time.Clock()
        self._render()

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick( self.metadata["render_fps"] )
        elif self.render_mode == "rgb_array":
            return np.transpose( 
                np.array(
                    pygame.surfarray.pixels3d( self.window_surface ) 
                ), 
                axes = ( 1, 0, 2 ) 
            )

    def close(self):
        if self.window_surface is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
        self.rsim.stop()

    def norm_pos( self, pos ):
        return np.clip(pos / self.max_pos, -self.NORM_BOUNDS, self.NORM_BOUNDS)

    def norm_v( self, v ):
        return np.clip(v / self.max_v, -self.NORM_BOUNDS, self.NORM_BOUNDS)

    def norm_w( self, w ):
        return np.clip(w / self.max_w, -self.NORM_BOUNDS, self.NORM_BOUNDS)
    

    def _render( self ):
        # Cria uma Surface para desenhar e ajusta no centro da tela 
        self.window_surface.fill( COLORS["GRAY"] )

        match_surface = pygame.surface.Surface( 
            self.field_size, 
            flags = pygame.SRCALPHA, 
            depth = 32 
        )

        # Computa e desenha o Heatmap no campo 
        self.compute_heatmap( )

        # Desenha o HeatMap 
        self._draw_heatmap( match_surface )

        # self._draw_grid( match_surface, spacing = 0.05, point_radius = 1 ) 
        ball = Ball(
            *self.pos_transform(self.frame.ball.x, self.frame.ball.y),
            self.field_renderer.scale
        )

        for i in range(self.n_robots_blue):
            robot = self.frame.robots_blue[i]
            x, y = self.pos_transform(robot.x, robot.y)
            rbt = VSSRobot(
                x,
                y,
                robot.theta,
                self.field_renderer.scale,
                robot.id,
                COLORS["BLUE"],
            )
            rbt.draw(match_surface)

        for i in range(self.n_robots_yellow):
            robot = self.frame.robots_yellow[i]
            x, y = self.pos_transform(robot.x, robot.y)
            rbt = VSSRobot(
                x,
                y,
                robot.theta,
                self.field_renderer.scale,
                robot.id,
                COLORS["YELLOW"],
            )
            rbt.draw(match_surface)
        ball.draw(match_surface)

        # Desenha o PathPlanning 
        self.compute_path_planning( match_surface )

        
        # Coordenadas do centro da tela
        x = int( (self.window_size[0]/2) - (self.field_size[0] / 2) )
        y = int( (self.window_size[1]/2) - (self.field_size[1] / 2) )

        # Desenha os elementos na tela 
        self.window_surface.blit( self.static_field_surface, ( x, y ) ) 
        self.window_surface.blit( match_surface, ( x, y ) )
        


    def _draw_trajectories(self):
        for i in range(self.n_robots_blue):
            rob = self.frame.robots_blue[i]
            if isinstance(rob, Robot):
                pts = rob.trajectory.points
                goals = rob.trajectory.goals
                if len(pts) > 1:
                    # converte lista de Point para lista de tuplas de pixels
                    screen_pts = [ self.pos_transform(p.x, p.y) for p in pts ]
                    pygame.draw.lines( self.window_surface, COLORS["BLUE"], False, screen_pts, 2 )
                if len(goals) > 0:
                    for i in range(len(rob.trajectory.goals)):
                        g = rob.trajectory.goals[i]
                        x,y = self.pos_transform(g.x, g.y)
                        # desenha um “X” verde
                        pygame.draw.circle( self.window_surface, COLORS["RED"], (x,y), 5, 2)




    def _compute_grid_centers( self, spacing: float = 0.1 ) -> np.ndarray:
        """
        Calcula a matriz de pontos no centro de cada célula do grid.
        Retorna um array shape=( n_rows, n_cols, 2 ) com coordenadas (x, y).
        """
        half_length = self.field.length / 2
        half_width  = self.field.width  / 2
        # Pontos ( X, Y ) deslocados meio espaçamento
        x_centers = np.arange(-half_length + spacing/2, half_length, spacing)
        y_centers = np.arange(-half_width  + spacing/2, half_width,  spacing)
        # Cria uma malha 2D de centros
        Xc, Yc = np.meshgrid(x_centers, y_centers)
        self.centers = np.stack([Xc, Yc], axis=-1)
        self.centers_Ui = np.full(
            ( len(y_centers), len(x_centers), 4 ),
            fill_value = [ 0, 0, 0, 127 ],
            dtype = np.uint8
        )
        return self.centers
    

    def compute_heatmap( self ):
        '''
            Cria um mapa de campo potencial baseado na presença das 
            forças potenciais de cada obstaculo em campo.
            
            1. Cria um mapa de calor baseado em self.compute_potencial_field( (x,y) )
            preenchendo células de tamanho `cell_px` com cores interpoladas.
            
            2. Mapeia o u∈[-1,1] num gradiente:
            Verde ( u = -1 )    Ganho por percorrer o caminho
            Amarelo ( u = 0 )   Custo zero 
            Vermelho ( u = +1 ) Caminho custoso 
        '''
        # prepara arrays de parâmetros
        if self.n_robots_blue > 1: 
            blues = [self.frame.robots_blue[i] for i in range(1, self.n_robots_blue)]
        else:
            blues = []
        if self.n_robots_yellow > 1:
            yellows = [self.frame.robots_yellow[i] for i in range(self.n_robots_yellow) ]
        else: 
            yellows = []
        robots = blues + yellows
        
        # Monta os arrays de parametros para usar o NJIT
        rx  = np.array([r.x for r in robots], dtype=np.float64)
        ry  = np.array([r.y for r in robots], dtype=np.float64)
        rvx = np.array([r.v_x for r in robots], dtype=np.float64)
        rvy = np.array([r.v_y for r in robots], dtype=np.float64)
        rth = np.array([r.theta for r in robots], dtype=np.float64)
        rvt = np.array([r.v_theta for r in robots], dtype=np.float64)
        rA  = np.array([r.force_field.A*10 for r in robots], dtype=np.float64)
        rs2 = np.array([r.force_field.sigma2 for r in robots], dtype=np.float64)
        rb  = np.array([r.force_field.beta for r in robots], dtype=np.float64)
        rg  = np.array([r.force_field.gamma for r in robots], dtype=np.float64)
        rk  = np.array([r.force_field.kappa for r in robots], dtype=np.float64)
        rom = np.array([r.force_field.omega_max for r in robots], dtype=np.float64)
        k_s = np.array([r.force_field.k_stretch for r in robots], dtype=np.float64)
        rlm = np.array([r.force_field.v_lin_max for r in robots], dtype=np.float64)
        self.centers_Ui = _compute_heatmap_core(
            self.centers, 
            self.frame.ball.x,
            self.frame.ball.y,
            self.frame.ball.v_x,
            self.frame.ball.v_y,
            self.frame.ball.theta,
            self.frame.ball.v_theta,
            rx, ry, rvx, rvy, rth, rvt,
            rA, rs2, rb, 0.2,
            rg, rk, rom, rlm, k_s,
            ( (3.0*robots[0].force_field.sigma)**2 )
        )

    def _draw_heatmap( self, screen: pygame.Surface ):
        screen_w, screen_h = screen.get_size()
        screen_w -= 2*self.field_renderer.goal_depth
        screen_h -= 2*self.field_renderer.margin
        n_rows, n_cols, _ = self.centers_Ui.shape
        # 1) Cria surface “pequena” de cells
        small = pygame.Surface((n_cols, n_rows), flags=pygame.SRCALPHA, depth=32)
        # 2) Copia todos os RGB de uma só vez
        #    surfarray.pixels3d entrega view (W×H×3) para escrever em C
        rgb_array = pygame.surfarray.pixels3d(small)
        # centers_Ui está em shape (n_rows,n_cols,4) → swapaxes para (n_cols,n_rows,4)
        # e fatiamos os 3 canais
        rgb_array[:, :, :] = self.centers_Ui.swapaxes(0,1)[:,:,:3]
        # e o alpha:
        alpha_array = pygame.surfarray.pixels_alpha(small)
        alpha_array[:, :] = self.centers_Ui.swapaxes(0,1)[:,:,3]
        # 3) Escala para o tamanho da tela em C
        #    smoothscale pode dar visual mais suave, mas scale é mais rápido
        large = pygame.transform.scale(small, (screen_w, screen_h))
        # 4) Blita no offset desejado
        offset = ( self.field_renderer.goal_depth, self.field_renderer.margin )
        screen.blit(large, offset)
    

    def compute_path_planning( self, screen: pygame.Surface ):
        p_start = (self.frame.robots_blue[0].x, self.frame.robots_blue[0].y )
        p_goal = ( self.frame.ball.x, self.frame.ball.y )
        w, h = self.field.width, self.field.length 
        self.path_planning_method = WeightedRRTPlanner( 
            p_start, p_goal, 
            w, h, 
            self,
        )
        self.path = self.path_planning_method.compute_path(  )
        self.draw_path( screen )


    def draw_path( self, screen, circle_radius = 5, line_width = 2 ):
        """
        Desenha o path planning considerando que os pontos estão em coordenadas
        centradas no campo:
        - x ∈ [-width/2, +width/2]
        - y ∈ [-height/2, +height/2]
        Converte para coordenadas de tela:
        sx = offset_x + (x + width/2) * scale
        sy = offset_y + (height/2 - y) * scale
        """
        if not self.path:
            return

        # Converte todos os pontos para coordenadas de tela
        screen_points = []
        offset = ( self.field_renderer.goal_depth, self.field_renderer.margin )
        for x, y in self.path:
            sx = offset[0] + (x + self.field.length/2) * self.field_renderer.scale
            sy = offset[1] + (self.field.width/2 + y) * self.field_renderer.scale
            screen_points.append((int(sx), int(sy)))

        # Desenha linhas conectando os pontos
        pygame.draw.lines(screen, COLORS['BLACK'], False, screen_points, line_width)

        # Desenha bolinhas nos nós do caminho
        for pt in screen_points:
            pygame.draw.circle(screen, COLORS['RED'], pt, circle_radius)
