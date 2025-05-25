from rSoccer.rsoccer_gym.Render.utils import COLORS
from screeninfo import get_monitors 

import numpy as np 
import pygame


def draw_rect_alpha( surface: pygame.Surface, color: pygame.Color, rect: pygame.Rect, thickness: int = 1 ):
    shape_surf = pygame.Surface( rect.size, pygame.SRCALPHA, depth = 32 )
    pygame.draw.rect( shape_surf, color, shape_surf.get_rect(), thickness )
    surface.blit(shape_surf, rect)

def draw_circle_alpha( surface: pygame.Surface, color: pygame.Color, center, radius):
    target_rect = pygame.Rect(center, (0, 0)).inflate((radius * 2, radius * 2))
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.circle(shape_surf, color, (radius, radius), radius)
    surface.blit(shape_surf, target_rect)

def draw_polygon_alpha( surface: pygame.Surface, color: pygame.Color, points):
    lx, ly = zip(*points)
    min_x, min_y, max_x, max_y = min(lx), min(ly), max(lx), max(ly)
    target_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.polygon(shape_surf, color, [(x - min_x, y - min_y) for x, y in points])
    surface.blit(shape_surf, target_rect)


class RenderField:
    goal_area_length: float = 0
    goal_area_width: float = 0

    grid_line_height: float = 1
    grid_line_width: float = 1
    grid_thrickness: float = 1
    grid_num_lines: int = 10
    grid_centers: np.ndarray = None
    grid: bool = False

    penalty_length: float = 0
    penalty_width: float = 0

    goal_width: float = 0
    goal_depth: float = 0
    
    center_circle_r: float = 0
    corner_arc_r: float = 0
    center_x: float = 0
    center_y: float = 0

    x_window_offset: float | int = 0
    y_window_offset: float | int = 0
    window_id: int = 0

    screen_width: float | int = 0
    screen_height: float | int = 0

    length: float | int = 0
    width: float | int = 0
    margin: int | float = 0
    _scale: int = 1

    grid_line_color : tuple = COLORS[ "BLACK" ]
    grid_centers_colors: tuple = COLORS[ "RED" ]

    # field_background_color: tuple = COLORS[ "FIELD_BG" ]
    field_background_color: tuple = COLORS[ "WINDOW_BG" ]
    field_line_color: tuple = COLORS[ "BLACK" ]


    def __init__(self, *args, **kwargs):    
        # Pega o tamanho da tela com preferencia para a segunda tela 
        screens = get_monitors()
        print( f"Number of screens: {len(screens)}")
        for screen in screens:
            print(f"Screen: {screen.name}, Width: {screen.width}, Height: {screen.height}")
        
        # Ajusta o Offset da janela para colocar na segunda tela 
        if len(screens) > 1:
            self.x_window_offset = -screens[0].x
            self.y_window_offset = -screens[0].y
            self.screen_width = screens[-1].width
            self.screen_height = screens[-1].height
            self.window_id = len(screens) -1 
            print(f"Using second screen: {screens[-1].name}")
        else:
            self.x_window_offset = 0
            self.y_window_offset = 0
            self.screen_width = screens[0].width
            self.screen_height = screens[0].height
            self.window_id = 0
            print(f"Using primary screen: {screens[0].name}")
        print(f"Window margin: ({self.x_window_offset}, {self.y_window_offset})")

        # Calcula a escla para fazer fullscreen
        self._scale_x = screens[-1].width / self.width
        self._scale_y = screens[-1].height / self.length
        if self._scale_x < self._scale_y:
            self._scale = self._scale_x
        else:
            self._scale = self._scale_y
        print(f"Scale: {self._scale}")


    @property
    def scale(self):
        return self._scale

    def _field_width(self):
        return self.length + 2 * self.margin

    def _field_height(self):
        return self.width + 2 * self.margin

    def _transform_params(self):
        self.length *= self._scale
        self.width *= self._scale
        self.penalty_length *= self._scale
        self.penalty_width *= self._scale
        self.goal_width *= self._scale
        self.goal_depth *= self._scale
        self.center_circle_r *= self._scale
        self.goal_area_length *= self._scale
        self.goal_area_width *= self._scale
        self.corner_arc_r *= self._scale

    def draw_background(self, screen):
        # screen.fill( self.background_color )
        pass 

    def draw_field_bounds( self, field ):
        back  = pygame.Rect( self.margin, self.margin, self.length, self.width )
        front = pygame.Rect( self.margin, self.margin, self.length, self.width )
        draw_rect_alpha( field, self.field_background_color, back, thickness = 0 )
        draw_rect_alpha( field, self.field_line_color, front, thickness = 2  )

    def draw_goal_left(self, field):
        back  = pygame.Rect( (self.margin - self.goal_depth), ((self.field_height - self.goal_width) // 2), self.goal_depth, self.goal_width+2 )
        front = pygame.Rect( self.margin - self.goal_depth, (self.field_height - self.goal_width) // 2, self.goal_depth, self.goal_width+2 )
        draw_rect_alpha( field, self.field_background_color, back, thickness = 0 )
        draw_rect_alpha( field, self.field_line_color, front, thickness = 2  )

    def draw_goal_right(self, field):
        back  = pygame.Rect( (self.field_width - self.margin), ((self.field_height - self.goal_width) // 2), self.goal_depth, self.goal_width-2 )
        front = pygame.Rect( self.field_width - self.margin, (self.field_height - self.goal_width) // 2, self.goal_depth, self.goal_width-2 )
        draw_rect_alpha( field, self.field_background_color, back, thickness = 0 )
        draw_rect_alpha( field, self.field_line_color, front, thickness = 2  )

    def draw_penalty_area_left(self, field):
        pygame.draw.rect(
            field,
            self.field_line_color,
            ( self.margin, (self.field_height - self.penalty_width) // 2, self.penalty_length, self.penalty_width ),
            2,
        )

    def draw_penalty_area_right(self, field):
        pygame.draw.rect(
            field,
            self.field_line_color,
            ( self.field_width - self.margin - self.penalty_length, (self.field_height - self.penalty_width) // 2, self.penalty_length, self.penalty_width ),
            2,
        )


    def draw_central_circle(self, screen):
        pygame.draw.circle(
            screen,
            self.field_line_color,
            (int(self.field_width / 2), int(self.field_height / 2)),
            int(self.center_circle_r),
            2,
        )

    def draw_central_line(self, screen):
        midfield_x = self.field_width / 2
        pygame.draw.line(
            screen,
            self.field_line_color,
            (midfield_x, self.margin),
            (midfield_x, self.field_height - self.margin),
            2,
        )

    def draw( self, screen: pygame.surface.Surface ) -> None:
        self.draw_background(screen)
        self.draw_field_bounds(screen)
        self.draw_central_line(screen)
        self.draw_central_circle(screen)
        self.draw_penalty_area_left(screen)
        self.draw_penalty_area_right(screen)
        self.draw_goal_left(screen)
        self.draw_goal_right(screen)


class VSSRenderField( RenderField ):
    center_circle_r = 0.2
    goal_area_length = 0
    goal_area_width = 0
    penalty_length = 0.15
    penalty_width = 0.7
    corner_arc_r = 0.01
    goal_width = 0.4
    goal_depth = 0.1
    
    margin = 0.1
    length = 1.5
    width = 1.3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define os parâmetros do campo baseados na escala dinamica 
        print( f"Length: {self.length}, Width: {self.width}, Margin: {self.margin}" )
        self.center_x = self.length / 2 + self.margin
        self.center_y = self.width / 2 + self.margin
        self.margin *= self._scale
        self.center_x *= self._scale
        self.center_y *= self._scale
        self._transform_params()

        # Aqui se pegava o tamanho necessário e desenhava a tela 
        self.field_width = self._field_width()
        self.field_height = self._field_height()
        self.window_size = ( int(self.screen_width), int(self.screen_height) )
        self.window_pos = ( int(self.x_window_offset), int(self.y_window_offset) )
        self.field_size = ( int(self.field_width), int(self.field_height) )
        print( f"Window size: {self.window_size} Window position: {self.window_pos}" )
        print( f"Field size: {self.field_size} Field center: ({self.center_x}, {self.center_y})" )


class Sim2DRenderField( RenderField ):
    length = 840.0
    width = 544.0
    margin = 40.0
    center_circle_r = 73.2
    penalty_length = 132.0
    penalty_width = 322.56
    goal_area_length = 44.0
    goal_area_width = 146.56
    goal_width = 112.16
    goal_depth = 19.52
    corner_arc_r = 8.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.field_width = self.length + 2 * self.margin
        self.field_height = self.width + 2 * self.margin
        self.window_size = (int(self.field_width), int(self.field_height))


class SSLRenderField(VSSRenderField):
    goal_area_length = 0
    goal_area_width = 0
    center_circle_r = 1
    penalty_length = 1
    penalty_width = 2
    corner_arc_r = 0.01
    goal_depth = 0.18
    goal_width = 1
    margin = 0.35
    length = 9
    width = 6
    _scale = 100


if __name__ == "__main__":
    field = Sim2DRenderField()
    pygame.display.init()
    pygame.display.set_caption("SSL Environment")
    window = pygame.display.set_mode(field.window_size)
    clock = pygame.time.Clock()
    while True:
        field.draw(window)
        pygame.event.pump()
        pygame.display.update()
        clock.tick(60)
