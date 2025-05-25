from rSoccer.rsoccer_gym.Render import COLORS, Ball, VSSRenderField, VSSRobot
from rSoccer.rsoccer_gym.Simulators.rsim import RSimVSS
from rSoccer.rsoccer_gym.Entities import Frame, Robot
from rSoccer.rsoccer_gym.Entities import PotentialField
from rSoccer.rsoccer_gym.Entities import Field

import gymnasium as gym
import numpy as np
import pygame
import time 

class VSSBaseEnv( gym.Env ):
    metadata = {
        "render.modes": [ "human", "rgb_array" ],
        "render_modes": [ "human", "rgb_array" ],
        "render_fps"  : 60,
        "render.fps"  : 60,
    }
    NORM_BOUNDS = 1.2
    u_var = 0.0

    def __init__(
        self,
        field_type: int,
        n_robots_blue: int,
        n_robots_yellow: int,
        time_step: float,
        render_mode = None,
        
        draw_grid: bool = True,
        grid_spacing: float = 0.01,
        grid_ratio: float = 0.5,

        draw_vector_field: bool = False,
        draw_heatmap: bool = False,
        force_field: PotentialField = None,
        field: Field = None,

        potential_amplitude: float = 1,
        potential_sigma_r: float = 0.1,
        potential_beta_v: float = 0.2 
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
        self.window_surface = None
        self.window_size = self.field_renderer.window_size
        self.field_size = self.field_renderer.field_size
        self.window_pos = self.field_renderer.window_pos
        self.window_id = self.field_renderer.window_id
        self.clock = None

        # self.draw_vector_field = draw_vector_field
        # self.draw_heatmap = draw_heatmap
        # self.force_field = force_field

        self.centers_colors = None 
        self.centers = None
        self.draw_grid = draw_grid
        self.grid_spacing = grid_spacing
        self.grid_centers = self._compute_grid_centers( spacing = self.grid_spacing )
        self.grid_ratio = grid_ratio

        self.potential_amplitude = potential_amplitude
        self.potential_sigma_r = potential_sigma_r
        self.potential_beta_v = potential_beta_v


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
        
        # Adiciona o campo potencial nos robos 
        for id in self.frame.robots_blue:
            self.frame.robots_blue[id].force_field.update_parameters( 
                self.potential_amplitude, 
                self.potential_sigma_r,
                self.potential_beta_v
            )
        for id in self.frame.robots_yellow:
            self.frame.robots_yellow[id].force_field.update_parameters( 
                self.potential_amplitude, 
                self.potential_sigma_r,
                self.potential_beta_v
            )
        
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
        
        t0 = time.time()
        # Desenha o Campo (linhas de campo, gols etc)
        self.field_renderer.draw( match_surface )
        t1 = time.time()

        # Computa o campo de relevo
        self.compute_heatmap( )
        t2 = time.time()

        # Desenha o Heatmap no campo 
        self._draw_heatmap( match_surface )
        
        # Desenha o Grid no campo 
        # self._draw_grid( match_surface, spacing = 0.05, point_radius = 1 ) 
        t3 = time.time()


        # Desenha vetores de força
        # self._draw_vector_field()
        
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
        
        # Desenha a superfície do jogo no centro da tela
        x = int( (self.window_size[0]/2) - (self.field_size[0] / 2) )
        y = int( (self.window_size[1]/2) - (self.field_size[1] / 2) )
        self.window_surface.blit( match_surface, ( x, y ) )
        t4 = time.time()
        
        print( f"Init counting time at {t0:10.4f}s" )
        print( f"dt compute heatMap: {(t2 - t1):10.4f}s" )
        print( f"dt field render: {(t1 - t0):10.4f}s" )
        print( f"dt Draw HeatMap: {(t3 - t2):10.4f}s" )
        print( f"dt draw players: {(t4 - t3):10.4f}s" )
        print( f"dt total: {(t4 - t0):10.4f}s" )


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


    def _draw_vector_field( self, step = 20, scale = 50 ):
        for i in range(0, self.window_size[0], step):
            for j in range(0, self.window_size[1], step):
                mx = (i - self.field_renderer.center_x) / self.field_renderer.scale
                my = (j - self.field_renderer.center_y) / self.field_renderer.scale
                fx, fy = self.force_field.force_at(mx, my)
                start = (i, j)
                end = (i + fx*scale, j - fy*scale)
                pygame.draw.line(self.window_surface, (0,0,255), start, end, 1)
                # opcionalmente desenhe a ponta da seta


    def _draw_grid( self, screen: pygame.surface.Surface, spacing: float = 0.1, point_radius: int = 2 ) -> None:
        """
        Desenha linhas de grid e um ponto em cada centro de célula.
        """
        if self.draw_grid: 
            half_length = self.field.length / 2
            half_width  = self.field.width  / 2

            # 1) Linhas verticais        
            x = -half_length
            while x <= half_length:
                start = self.pos_transform( x, -half_width)
                end   = self.pos_transform( x,  half_width)
                pygame.draw.line( screen, COLORS["GRID"], start, end, 1)
                x += spacing

            # 2) Linhas horizontais
            y = -half_width
            while y <= half_width:
                start = self.pos_transform(-half_length, y)
                end   = self.pos_transform( half_length, y)
                pygame.draw.line( screen, COLORS["GRID"], start, end, 1)
                y += spacing

            # 3) Pontos nos centros
            p_rows, p_cols, p_dim = self.centers.shape 
            for xi in range( p_rows ):
                for yi in range( p_cols ):
                    pxi, pyi = self.centers[ xi, yi ]
                    px, py = self.pos_transform( pxi, pyi )
                    pcolor = tuple(self.centers_colors[ xi, yi ]) 
                    pygame.draw.circle( screen, pcolor, (px, py), point_radius )



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
        self.centers_colors = np.full(
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
        # Pega a quantidade de pontos e divide pelo tamanho do Canva 
        n_rows, n_cols, n_dim = self.centers.shape
        # Computa cada pedacinho do canva 
        for i in range( n_rows ):
            for j in range( n_cols ):
                # Pega o ponto central de cada ponto do grid calculado
                mx, my = self.centers[i, j]
                # Avalia o campo normalizado e mapeia para cor
                u     = self._compute_potencial_field( mx, my )  # ∈[-1,1]
                color = self._compute_heatmap_color(u)           # (r,g,b,a)
                self.centers_colors[i,j] = color

    def _draw_heatmap( self, screen: pygame.Surface ):
        # Cria uma Superfície para servir de Canva para o HeatMap 
        screen_w, screen_h = screen.get_size()
        hm_surf = pygame.Surface( ( screen_w, screen_h ), flags = pygame.SRCALPHA )
        # Pega a quantidade de pontos e divide pelo tamanho do Canva 
        n_rows, n_cols, n_dim = self.centers.shape
        cell_w_px = screen_w // n_cols
        cell_h_px = screen_h // n_rows
        # Preenche cada pedacinho do canva 
        for i in range( n_rows ):
            for j in range( n_cols ):
                # Pixel de coordenada começando pelo canto superior-esquerdo do Grid 
                px0 = j * cell_w_px
                py0 = i * cell_h_px
                # Preenche o retângulo daquela célula
                rect = pygame.Rect( px0, py0, cell_w_px, cell_h_px )
                hm_surf.fill( self.centers_colors[i,j], rect )
        screen.blit(  hm_surf, ( 0.1*self.field_renderer.scale, 0.1*self.field_renderer.scale ) )

    def _compute_heatmap_color( self, u: float ) -> tuple[ int, int, int, int ]:
        if u < -0.1:
            t = min(u, 1.0)  # t ∈ [0,1]
            r = 0
            g = int( 255 * (1-t) ) 
            b = int( 255 * t)
        elif u <= 0.1:
            r = 0
            g = 255
            b = 0
        else:
            t = min(u, 1.0)  # t ∈ [0,1]
            r = int(255 * t)
            g = int( 255 * (1-t) ) 
            b = 0
        return ( r%256, g%256, b%256, 127)

    def _compute_potencial_field( self, xpos: int, ypos: int ) -> float:
        """
            Soma o potencial de todos os robôs no ponto (xpos, ypos):
            Obs: Temos que ignorar a influencia do robo em treinamento 
            Retorna U_total ∈ ℝ.
        """
        # Percorre robôs azuis, pulando o índice 0 (Agente)
        U_total: float = 0.0
        A_max: float = self.frame.robots_blue[0].force_field.A 

        if len(self.frame.robots_blue) > 1:
            for id in self.frame.robots_blue[1:]:
                potential = self.frame.robots_blue[id].potential_at(xpos, ypos)
                if potential > 0.0 : 
                    U_total += potential 
        if len(self.frame.robots_yellow) > 1:
            for id in self.frame.robots_yellow:
                potential = self.frame.robots_yellow[id].potential_at(xpos, ypos)
                if potential > 0.0: 
                    U_total += potential 
        if A_max == 0:
            return 0.0 
        else: 
            return U_total / A_max 