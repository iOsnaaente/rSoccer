import matplotlib.pyplot as plt
import random
import math

class Node:
    def __init__(self, point: tuple[float, float], parent: 'Node' = None):
        self.point = point
        self.parent = parent

class Circle:
    def __init__(self, center: tuple[float, float], radius: float):
        self.center = center
        self.radius = radius

def dist( p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def is_collision(p1: tuple[float, float], p2: tuple[float, float], obstacles: list[Circle]) -> bool:
    """Verifica colisão do segmento p1-p2 com qualquer círculo."""
    for obs in obstacles:
        cx, cy = obs.center
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        # Caso o ponto seja igual
        if dx == 0 and dy == 0:
            if dist(p1, obs.center) <= obs.radius:
                return True
            continue
        # Projeta o centro do círculo na reta
        t = ((cx - x1)*dx + (cy - y1)*dy) / (dx*dx + dy*dy)
        t = max(0.0, min(1.0, t))
        closest = (x1 + t*dx, y1 + t*dy)
        if dist(closest, obs.center) <= obs.radius:
            return True
    return False


class RRTPlanner:
    height: float
    width: float
    
    start: float
    goal: float
    
    sample_bias: float
    obstacles: float
    
    max_iter: float
    step_size: float

    def __init__(self,
        start: tuple[float, float], goal: tuple[float, float],
        width: float, height: float,
        obstacles: list[Circle],
        max_iter: int = 1000,
        step_size: float = 1.0,
        sample_bias: float = 0.05
    ):
        self.start = start
        self.goal = goal
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.max_iter = max_iter
        self.step_size = step_size
        self.sample_bias = sample_bias  

    def plan( self ) -> list | None :
        tree = [Node(self.start)]
        for _ in range(self.max_iter):
            if random.random() < self.sample_bias:
                rnd = self.goal
            else:
                rnd = (random.uniform(0, self.width), random.uniform(0, self.height))
            nearest = min(tree, key=lambda n: dist(n.point, rnd))
            theta = math.atan2(rnd[1] - nearest.point[1], rnd[0] - nearest.point[0])
            new_pt = (
                nearest.point[0] + self.step_size * math.cos(theta),
                nearest.point[1] + self.step_size * math.sin(theta)
            )
            # fora dos limites?
            if not (0 <= new_pt[0] <= self.width and 0 <= new_pt[1] <= self.height):
                continue
            # colisão?
            if is_collision(nearest.point, new_pt, self.obstacles):
                continue
            new_node = Node(new_pt, nearest)
            tree.append(new_node)
            # chegou perto do goal?
            if dist(new_pt, self.goal) <= self.step_size and not is_collision(new_pt, self.goal, self.obstacles):
                goal_node = Node(self.goal, new_node)
                tree.append(goal_node)
                # reconstrói o caminho
                path = []
                cur = goal_node
                while cur is not None:
                    path.append(cur.point)
                    cur = cur.parent
                return list(reversed(path)), tree
        return None


    def draw( self, path: list[Node] ) -> None:
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        # obstáculos
        for o in self.obstacles:
            c = plt.Circle(o.center, o.radius, color='gray', alpha=0.5)
            ax.add_patch(c)
        # árvore
        for n in tree:
            if n.parent:
                x0, y0 = n.point
                x1, y1 = n.parent.point
                ax.plot([x0, x1], [y0, y1], '-g', linewidth=0.5)
        # caminho
        if path:
            xs, ys = zip(*path)
            ax.plot(xs, ys, '-r', linewidth=2)
        # start/goal
        ax.plot(self.start[0], self.start[1], 'bs', label='Start')
        ax.plot(self.goal[0], self.goal[1], 'ms', label='Goal')
        ax.legend()
        ax.set_xlabel('X'); ax.set_ylabel('Y')
        ax.set_title(f'RRT (bias={self.sample_bias*100:.1f}%)')
        ax.grid(True)
        plt.show()


def generate_obstacles(
    num: int,
    width: float, height: float,
    start: tuple[float,float], goal: tuple[float,float],
    radius_range: tuple[float,float] 
) -> list[Circle]:
    obs = []
    while len(obs) < num:
        r = random.uniform(*radius_range)
        x = random.uniform(r, width - r)
        y = random.uniform(r, height - r)
        if dist((x,y), start) <= r or dist((x,y), goal) <= r:
            continue
        if any(dist((x,y), o.center) <= r + o.radius for o in obs):
            continue
        obs.append(Circle((x,y), r))
    return obs


if __name__ == '__main__':
    W, H = 100.0, 100.0
    start = (0.0, 0.0)
    goal  = (
        random.uniform( 45.0,90.0 ), 
        random.uniform( 45.0,90.0 )
    )
    obstacles = generate_obstacles( 10, W, H, start, goal, (5.0, 10.0) )

    planner = RRTPlanner(
        start, goal, 
        W, H, 
        obstacles,
        max_iter = 5000, step_size = 2.0,
        sample_bias = 0.5
    ) 
    path, tree = planner.plan()
    planner.draw( path )
