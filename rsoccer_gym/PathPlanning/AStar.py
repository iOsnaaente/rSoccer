import math
import heapq
import numpy as np
from rSoccer.rsoccer_gym.Entities.PotentialField import _potential_core

class AStarNode:
    def __init__(self, position, g=math.inf, h=0.0, parent=None):
        self.position = position  # tuple (x, y)
        self.g = g                # custo acumulado
        self.h = h                # heurística
        self.f = g + h            # custo total
        self.parent = parent      # nó anterior

    def __lt__(self, other):
        return self.f < other.f

# Distância Euclidiana
distance = lambda p1, p2: math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# compute_point_potential(x, y, env) segue igual ao código anterior
# Presume-se que compute_point_potential foi definido ou importado acima

class WeightedAStarPlanner:
    def __init__(
        self,
        start,
        goal,
        width,
        height,
        env,
        step_size,
        potential_threshold=math.inf
    ):
        """
        Parâmetros:
        - start, goal: (x, y) em metros
        - width, height: dimensões do campo
        - env: instância VSSS_Env para cálculo de potenciais
        - step_size: distância entre vizinhos
        - potential_threshold: descarta vizinhos com U > limiar
        """
        self.start = start
        self.goal = goal
        self.width = width
        self.height = height
        self.env = env
        self.step = step_size
        self.pot_thresh = potential_threshold

    def get_neighbors(self, pos):
        """Gera 8 vizinhos ao passo definido, dentro dos limites do campo e abaixo do limiar de potencial."""
        x0, y0 = pos
        dirs = [
            (1,0),(-1,0),(0,1),(0,-1),
            (1,1),(1,-1),(-1,1),(-1,-1)
        ]
        neighbors = []
        for dx, dy in dirs:
            nx = x0 + dx * self.step
            ny = y0 + dy * self.step
            # dentro do campo
            if -self.width/2 <= nx <= self.width/2 and -self.height/2 <= ny <= self.height/2:
                U = compute_point_potential(nx, ny, self.env)
                if U <= self.pot_thresh:
                    cost = distance((x0,y0),(nx,ny)) * (1.0 + U)
                    neighbors.append(((nx,ny), cost))
        return neighbors

    def reconstruct_path(self, node):
        path = []
        cur = node
        while cur:
            path.append(cur.position)
            cur = cur.parent
        return list(reversed(path))

    def compute_path_planning(self):
        """Execução do A* ponderado pelo campo de potenciais."""
        open_set = []
        start_node = AStarNode(self.start, g=0.0,
                               h=distance(self.start, self.goal),
                               parent=None)
        heapq.heappush(open_set, start_node)
        came_from = {}
        g_score = {self.start: 0.0}

        visited = set()
        while open_set:
            current = heapq.heappop(open_set)
            if current.position in visited:
                continue
            # Critério de sucesso: próximo o bastante do objetivo
            if distance(current.position, self.goal) <= self.step:
                return self.reconstruct_path(current)
            visited.add(current.position)

            for nbr_pos, cost in self.get_neighbors(current.position):
                tentative_g = current.g + cost
                if tentative_g < g_score.get(nbr_pos, math.inf):
                    g_score[nbr_pos] = tentative_g
                    h = distance(nbr_pos, self.goal)
                    nbr_node = AStarNode(nbr_pos, g=tentative_g, h=h, parent=current)
                    heapq.heappush(open_set, nbr_node)
        return None

# Exemplo de uso:
# planner = WeightedAStarPlanner(
#     start=p_start,
#     goal=p_goal,
#     width=env.field.width,
#     height=env.field.height,
#     env=env,
#     step_size=0.1,
#     potential_threshold=0.5
# )
# path = planner.compute_path_planning()
