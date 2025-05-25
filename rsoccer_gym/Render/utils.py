import pygame 

# COLORS RGB
COLORS = {
    "BLACK":        pygame.Color(   0,   0,   0, a = 255 ),
    "WHITE":        pygame.Color( 220, 220, 220, a = 255 ),
    "GRAY":         pygame.Color( 200, 200, 200, a = 255 ),
    "BG_GREEN":     pygame.Color(  20,  90,  45, a = 255 ),
    "GREEN":        pygame.Color(   0, 128,   0, a = 255 ),
    "ROBOT_BLACK":  pygame.Color(  25,  25,  25, a = 255 ),
    "ORANGE":       pygame.Color( 253, 106,   2, a = 255 ),
    "BLUE":         pygame.Color(   0,  64, 255, a = 255 ),
    "YELLOW":       pygame.Color( 250, 218,  94, a = 255 ),
    "GREEN":        pygame.Color(  57, 220,  20, a = 255 ),
    "RED":          pygame.Color( 151,  21,   0, a = 255 ),
    "PURPLE":       pygame.Color( 102,  51, 153, a = 255 ),
    "PINK":         pygame.Color( 220,   0, 220, a = 255 ),

    "WINDOW_BGA":   pygame.Color( 255, 255, 255, a = 127 ),
    "WINDOW_BG":    pygame.Color( 255, 255, 255, a = 255 ),

    "FIELD_BGA":    pygame.Color(  20,  90,  45, a = 127 ),
    "FIELD_BG":     pygame.Color(  20,  90,  45, a = 255 ),
    
    "GRID":         pygame.Color( 200, 200, 200, a = 255 ),
    "GRID_DARK":    pygame.Color( 100, 100, 100, a = 255 ),
    
    "POINTS":       pygame.Color( 255,  50,  50, a = 255 ),
    "POINTS_DARK":  pygame.Color( 200,  25,  25, a = 255 ),
}


TAG_ID_COLORS = {
    0: {
        0: COLORS["PINK"],
        1: COLORS["GREEN"],
        2: COLORS["PINK"],
        3: COLORS["PINK"],
    },
    1: {
        0: COLORS["GREEN"],
        1: COLORS["GREEN"],
        2: COLORS["PINK"],
        3: COLORS["PINK"],
    },
    2: {
        0: COLORS["GREEN"],
        1: COLORS["GREEN"],
        2: COLORS["PINK"],
        3: COLORS["GREEN"],
    },
    3: {
        0: COLORS["PINK"],
        1: COLORS["GREEN"],
        2: COLORS["PINK"],
        3: COLORS["GREEN"],
    },
    4: {
        0: COLORS["PINK"],
        1: COLORS["PINK"],
        2: COLORS["GREEN"],
        3: COLORS["PINK"],
    },
    5: {
        0: COLORS["GREEN"],
        1: COLORS["PINK"],
        2: COLORS["GREEN"],
        3: COLORS["PINK"],
    },
    6: {
        0: COLORS["GREEN"],
        1: COLORS["PINK"],
        2: COLORS["GREEN"],
        3: COLORS["GREEN"],
    },
    7: {
        0: COLORS["PINK"],
        1: COLORS["PINK"],
        2: COLORS["GREEN"],
        3: COLORS["GREEN"],
    },
    8: {
        0: COLORS["GREEN"],
        1: COLORS["GREEN"],
        2: COLORS["GREEN"],
        3: COLORS["GREEN"],
    },
    9: {
        0: COLORS["PINK"],
        1: COLORS["PINK"],
        2: COLORS["PINK"],
        3: COLORS["PINK"],
    },
    10: {
        0: COLORS["PINK"],
        1: COLORS["GREEN"],
        2: COLORS["GREEN"],
        3: COLORS["PINK"],
    },
    11: {
        0: COLORS["GREEN"],
        1: COLORS["PINK"],
        2: COLORS["PINK"],
        3: COLORS["GREEN"],
    },
    12: {
        0: COLORS["GREEN"],
        1: COLORS["GREEN"],
        2: COLORS["GREEN"],
        3: COLORS["PINK"],
    },
    13: {
        0: COLORS["GREEN"],
        1: COLORS["PINK"],
        2: COLORS["PINK"],
        3: COLORS["PINK"],
    },
    14: {
        0: COLORS["PINK"],
        1: COLORS["GREEN"],
        2: COLORS["GREEN"],
        3: COLORS["GREEN"],
    },
    15: {
        0: COLORS["PINK"],
        1: COLORS["PINK"],
        2: COLORS["PINK"],
        3: COLORS["GREEN"],
    },
}
