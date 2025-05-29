import random
import math


"""Retorna um ponto aleatório dentro dos limites do campo."""
sample_random_point = lambda width, height : (
    random.uniform(0, width), random.uniform(0, height)
)

"""Distância Euclidiana entre dois pontos."""
distance = lambda p1, p2 : math.hypot( p2[0] - p1[0], p2[1] - p1[1] )

"""Encontra o nó mais próximo na árvore em relação a um ponto."""
nearest = lambda nodes, point : min( nodes, key = lambda node: distance( node.position, point ) )


class Node:
    def __init__(self, position, parent=None, cost=0.0):
        self.position = position  # tuple (x, y)
        self.parent = parent      # Node
        self.cost = cost          # float: custo acumulado desde a raiz


def steer( from_node, to_point, step_size ):
    """Cria um novo nó a partir de from_node na direção de to_point, limitado por step_size."""
    from_pos = from_node.position
    dx = to_point[0] - from_pos[0]
    dy = to_point[1] - from_pos[1]
    dist = math.hypot(dx, dy)
    if dist <= step_size:
        new_pos = to_point
    else:
        theta = math.atan2(dy, dx)
        new_pos = (from_pos[0] + step_size * math.cos(theta),
                   from_pos[1] + step_size * math.sin(theta))
    return Node(new_pos, parent=from_node)


def collides(point, obstacles):
    """Verifica se um ponto está dentro de algum obstáculo circular."""
    for (ox, oy, r) in obstacles:
        if distance(point, (ox, oy)) <= r:
            return True
    return False


class RRTPlanner:

    """
        Método de RRT. Os valores de entrada da classe são: 
            start: tuple(x, y) -> ponto inicial do caminho.
            goal: tuple(x, y) -> ponto final desejado.
            width: float -> largura do campo de planejamento.
            height: float -> altura do campo de planejamento.
            step_size: float -> distância máxima entre nós consecutivos.
            max_iter: int -> número máximo de iterações para busca.
            n_obstacles: int -> número de obstáculos circulares a serem gerados.
            obs_radius_range: tuple(float, float) -> intervalo de raios dos obstáculos.
    """
    def __init__( self, 
        start, goal, 
        width, height, 
        step_size = 10.0,
        max_iter = 1000,
        n_obstacles = 5,
        obs_radius_range = (5, 15),
        neighbor_radius = 20.0
    ):

        self.start = Node(start, cost=0.0)
        self.goal = Node(goal, cost=float('inf'))
        self.width = width
        self.height = height
        self.step_size = step_size
        self.max_iter = max_iter
        self.neighbor_radius = neighbor_radius
        self.tree = []

        # Geração de obstáculos aleatórios (círculos)
        self.obstacles = []
        for _ in range(n_obstacles):
            while True:
                r = random.uniform( *obs_radius_range)
                x = random.uniform( r, width - r)
                y = random.uniform( r, height - r)

                # verifica colisão com obstáculos existentes e com o goal
                if not collides((x, y), self.obstacles) and distance((x, y), self.goal.position) > r:
                    self.obstacles.append((x, y, r))
                    break


    def get_near_nodes(self, new_node, nodes):
        """Retorna lista de nós em nodes dentro de neighbor_radius de new_node."""
        return [
            node for node in nodes
                if distance(node.position, new_node.position) <= self.neighbor_radius]

    def compute_path_planning(self):
        """Executa o RRT*, retorna caminho ou None."""
        tree = [self.start]
        for _ in range(self.max_iter):
            rand_pt = sample_random_point(self.width, self.height)
            if collides(rand_pt, self.obstacles):
                continue
            # Encontra nó mais próximo e tenta estender
            nearest_node = nearest(tree, rand_pt)
            new_node = steer(nearest_node, rand_pt, self.step_size)
            if collides(new_node.position, self.obstacles):
                continue
            # Rewire: encontra vizinhos antes de adicionar
            neighbors = self.get_near_nodes(new_node, tree)
            # Escolhe melhor pai (menor custo)
            min_cost = new_node.cost
            best_parent = new_node.parent
            for nb in neighbors:
                tentative_cost = nb.cost + distance(nb.position, new_node.position)
                if tentative_cost < min_cost:
                    best_parent = nb
                    min_cost = tentative_cost
            new_node.parent = best_parent
            new_node.cost = min_cost
            tree.append(new_node)

            # Rewire vizinhos
            for nb in neighbors:
                rewire_cost = new_node.cost + distance(new_node.position, nb.position)
                if rewire_cost < nb.cost:
                    nb.parent = new_node
                    nb.cost = rewire_cost
                    
            # Verifica conexão ao objetivo
            if distance(new_node.position, self.goal.position) < self.step_size and \
               not collides(self.goal.position, self.obstacles):
                self.goal.parent = new_node
                self.goal.cost = new_node.cost + distance(new_node.position, self.goal.position)
                return self._reconstruct_path(self.goal)
        return None

    def _reconstruct_path(self, node):
        path = []
        current = node
        while current:
            path.append(current.position)
            current = current.parent
        path.reverse()
        return path


if __name__ == "__main__":
    from matplotlib.patches import Circle
    import matplotlib.pyplot as plt

    while True:

        planner = RRTPlanner(
            start=(0, 0), 
            goal=( random.uniform(45, 90), random.uniform(45, 90) ), 
            width=100,  height=100,
            step_size = 5, 
            max_iter=10000, 
            n_obstacles=5, 
            obs_radius_range=(5, 15)
        )
        path = planner.compute_path_planning()

        # Plot setup
        fig, ax = plt.subplots()
        # Desenha obstáculos
        for (ox, oy, r) in planner.obstacles:
            circle = Circle((ox, oy), r, color='red', alpha=0.3)
            ax.add_patch(circle)

        # Caminho
        if path:
            xs = [pt[0] for pt in path]
            ys = [pt[1] for pt in path]
            ax.plot(xs, ys, marker='o', linestyle='-', label='Caminho RRT')
        else:
            print("Falha em encontrar caminho dentro do limite de iterações.")

        # Pontos início e objetivo
        ax.scatter([planner.start.position[0]], [planner.start.position[1]], s=100, marker='s', label='Início')
        ax.scatter([planner.goal.position[0]], [planner.goal.position[1]], s=100, marker='*', label='Objetivo')

        # Configurações finais do plot
        ax.set_xlim(0, planner.width)
        ax.set_ylim(0, planner.height)
        ax.set_title('RRT Path Planning com Obstáculos Circulares')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True)

        plt.show()
