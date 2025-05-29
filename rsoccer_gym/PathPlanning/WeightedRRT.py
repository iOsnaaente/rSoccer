from numba import njit

from rSoccer.rsoccer_gym.Entities.PotentialField import _potential_core
import matplotlib.pyplot as plt
import numpy as np
import random
import math


"""
    Weighted RRT Planner — Hiperparâmetros de Configuração
    step_size: float (padrão=0.05)
        • Define o comprimento máximo de cada expansão da árvore.
        • Valores menores geram passos mais finos: melhor adaptação a curvas e pequenos obstáculos,
        mas aumentam o número de nós e o custo computacional.
        • As vezes falha com valores pequenos.
        • Valores maiores aceleram a cobertura do espaço, porém podem “pular” zonas estreitas
        e reduzir a suavidade do caminho.

    max_iter: int (padrão=100)
        • Número máximo de iterações (tentativas de expansão) antes de abortar.
        • Maior max_iter aumenta a probabilidade de encontrar caminhos em espaços complexos,
        mas alonga o tempo de execução.
        • Reduzir acelera retornos rápidos quando o ambiente é simples ou o tempo é restrito.
        • Este é o ponto mais crítico, pois: 
            - Valores grandes darão melhores resultados, mas a um custo de tempo de execução muito alto.
            - Valores pequenso podem falhar ou dar resultados ruins, mas rapidamente 

    sample_bias: int (padrão=20)
        • Quantidade de pontos candidatos a cada iteração para seleção ponderada.
        • Bias maior melhora a “inteligência” da amostragem (melhorá qualidade do caminho),
        pois considera mais opções, mas eleva custo por iteração.
        • Bias menor acelera cada passo, mas tende a caminhos de qualidade variável/aleatória.

    alpha: float (padrão=5.0)
        • Temperatura da distribuição de amostragem exp(-α·U).
        • α alto concentra amostras em regiões de baixo potencial (mais exploração guiada),
        porém pode ficar preso em mínimos locais.
        • α baixo uniformiza mais a amostragem (exploração global), mas perde o viés
        de evitar zonas “caras”.

    goal_sample_rate: float (padrão=0.1)
        • Probabilidade de amostrar diretamente o ponto-objetivo em cada iteração.
        • Valores mais altos aceleram a convergência ao goal, mas reduzem exploração
        de zonas alternativas.
        • Valores mais baixos exploram melhor o espaço, porém podem atrasar a descoberta
        do goal, sobretudo em ambientes abertos.

    k_att: float (padrão=5.0)
        • Ganho aplicado ao potencial atrativo (bola ou goal) na função de custo.
        • k_att alto faz o planner “puxar” mais fortemente para regiões atrativas,
        encurtando trajetos quase retos, mas pode ignorar pequenas repulsões.
        • k_att baixo reduz essa atração, priorizando desvios suaves mesmo perto do objetivo.

    k_rep: float (padrão=1.0)
        • Ganho aplicado ao potencial repulsivo (robôs/oponentes) na função de custo.
        • k_rep alto penaliza fortemente regiões de alto U, produzindo trajetos mais seguros,
        porém possivelmente mais longos.
        • k_rep baixo tolera incursões em zonas de repulsão para encurtar distância,
        ideal em cenários onde pequenas infiltrações são aceitáveis.
"""


@njit( cache = False )
def compute_potential_jit(
    # Função JIT-otimizada para calcular potencial numérico
    goal_x: float, goal_y: float,
    robots_x: np.ndarray, robots_y: np.ndarray,
    robots_vx: np.ndarray, robots_vy: np.ndarray,
    robots_th: np.ndarray, robots_vt: np.ndarray,
    robots_A:  np.ndarray, robots_s2: np.ndarray, robots_b:  np.ndarray,
    robots_g:  np.ndarray, robots_k:  np.ndarray, robots_om: np.ndarray,
    robots_ks:  np.ndarray, robots_lm: np.ndarray,
    ball_x: float, ball_y: float,
    ball_vx: float, ball_vy: float,
    ball_theta: float, ball_vtheta: float,
    sigma: float, A_max: float
) -> float:
    inf2 = (3.0 * sigma) ** 2
    U_total = 0.0
    n = robots_x.shape[0]
    # Repulsão dos robôs
    for i in range(n):
        dx = goal_x - robots_x[i]
        dy = goal_y - robots_y[i]
        if dx * dx + dy * dy <= inf2:
            U = _potential_core(
                dx, dy,
                robots_vx[i], robots_vy[i],
                robots_th[i], robots_vt[i],
                robots_A[i],  robots_s2[i], robots_b[i],
                1e-5,
                robots_g[i], robots_k[i], robots_om[i],
                robots_ks[i], robots_lm[i]
            )
            if U > 0.0:
                U_total += U
    # Atração da bola
    dx = goal_x - ball_x
    dy = goal_y - ball_y
    if dx * dx + dy * dy <= inf2:
        Ub = _potential_core(
            dx, dy,
            ball_vx, ball_vy,
            ball_theta, ball_vtheta,
            2.50, (0.075*0.075), 10.0,      # A, sigma2, beta
            1e-5,                           # Epsilon 
            1e-3, 25e-3,                    # Gamma e Kappa 
            45,                             # Omega Max 
            1.0, 1.0,                       # K_stretch e V_lin_max
        )
        if Ub > 0.0:
            U_total -= Ub

    # Normalização e clamp
    u = U_total / A_max
    if u > 1.0:
        return 1.0
    if u < -1.0:
        return -1.0
    return u


# Distância Euclidiana
def distance( p1: tuple[float, float], p2: tuple[float, float] ) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


class Node:
    def __init__( self, 
        position: tuple[float, float], 
        parent: 'Node' = None, 
        cost: float = 0.0 
    ):
        self.position = position  # tupla (x, y)
        self.parent = parent      # Node anterior
        self.cost = cost          # custo acumulado até este nó (não usado aqui)


class WeightedRRTPlanner:
    def __init__(self,
                 start: tuple[float, float], goal: tuple[float, float],
                 width: float, height: float,
                 env: 'VSSS_Env', # type: ignore
                 step_size: float = 0.05,
                 max_iter: int = 100,
                 sample_bias: int = 20,
                 alpha: float = 5.0,
                 goal_sample_rate: float = 0.1,
                 k_att=5.0,
                 k_rep=1.0,
    ):
        # Configurações iniciais
        self.start = start
        self.goal = goal

        self.width = width
        self.height = height

        self.env = env

        self.step_size = step_size
        self.max_iter = max_iter
        self.sample_bias = sample_bias

        self.alpha = alpha
        self.k_att = k_att
        self.k_rep = k_rep
        self.goal_sample_rate = goal_sample_rate

        # Vetores de parâmetros
        if self.env.n_robots_blue > 1:
            robots = list(self.env.frame.robots_blue.values())[1:] + list(self.env.frame.robots_yellow.values())
        else:
            robots = list(self.env.frame.robots_yellow.values())
        self.robots_x  = np.array( [r.x        for r in robots], dtype = np.float64 )
        self.robots_y  = np.array( [r.y        for r in robots], dtype = np.float64 )
        self.robots_vx = np.array( [r.v_x      for r in robots], dtype = np.float64 )
        self.robots_vy = np.array( [r.v_y      for r in robots], dtype = np.float64 )
        self.robots_th = np.array( [r.theta    for r in robots], dtype = np.float64 )
        self.robots_vt = np.array( [r.v_theta  for r in robots], dtype = np.float64 )
        self.robots_A  = np.array( [r.force_field.A        for r in robots], dtype = np.float64 )
        self.robots_s2 = np.array( [r.force_field.sigma2   for r in robots], dtype = np.float64 )
        self.robots_b  = np.array( [r.force_field.beta     for r in robots], dtype = np.float64 )
        self.robots_g  = np.array( [r.force_field.gamma    for r in robots], dtype = np.float64 )
        self.robots_k  = np.array( [r.force_field.kappa    for r in robots], dtype = np.float64 )
        self.robots_om = np.array( [r.force_field.omega_max for r in robots], dtype = np.float64 )
        self.robots_ks  = np.array([r.force_field.k_stretch for r in robots], dtype = np.float64 )
        self.robots_lm = np.array([r.force_field.v_lin_max for r in robots], dtype = np.float64 )
        
        # Parâmetros da bola
        b = env.frame.ball
        self.ball_x, self.ball_y = float(b.x), float(b.y)
        self.ball_vx, self.ball_vy = float(b.v_x), float(b.v_y)
        self.ball_theta, self.ball_vtheta = float(b.theta), float(b.v_theta)
        
        # Parametros gerais 
        self.sigma = float( robots[0].force_field.sigma )
        self.A_max = max( self.robots_A.max(), 2.5)

        # Árvore de nós
        self.tree = [ Node( self.start ) ]


    def compute_potential( self, x: float, y: float ):
        """ Versão JIT do método """  
        return compute_potential_jit(
            x, y,
            self.robots_x, self.robots_y, self.robots_vx, self.robots_vy,
            self.robots_th, self.robots_vt,
            self.robots_A, self.robots_s2, self.robots_b,
            self.robots_g, self.robots_k, self.robots_om,
            self.robots_ks, self.robots_lm,
            self.ball_x, self.ball_y,
            self.ball_vx, self.ball_vy,
            self.ball_theta, self.ball_vtheta,
            self.sigma, self.A_max
        )


    def sample_point(self):
        # % de chance de samplear o goal
        if random.random() < self.goal_sample_rate:
            return self.goal
        # Amostragem ponderada por U
        cands, Us = [], []
        for _ in range(self.sample_bias):
            x = random.uniform(-self.width/2, self.width/2)
            y = random.uniform(-self.height/2, self.height/2)
            U = self.compute_potential(x, y)
            cands.append((x, y))
            Us.append(U)
        weights = np.exp(-self.alpha * np.array(Us))
        weights /= weights.sum()
        idx = np.random.choice( len(cands), p = weights )
        return cands[idx]
    

    def nearest(self, pt):
        """ Usa custo acumulado + distância ponderada pelo potencial no próprio nó """ 
        def metric(node):
            U_node = self.compute_potential(*node.position)
            factor = 1 + (self.k_rep * max(U_node, 0) + self.k_att * min(U_node, 0))
            return node.cost + distance(node.position, pt) * factor
        return min(self.tree, key=metric)

    def steer(self, from_node, to_point):
        fx, fy = from_node.position
        tx, ty = to_point
        dx, dy = tx - fx, ty - fy
        d = math.hypot(dx, dy)
        # Move até step_size
        if d <= self.step_size:
            nx, ny = tx, ty
        else:
            ang = math.atan2(dy, dx)
            nx = fx + self.step_size * math.cos(ang)
            ny = fy + self.step_size * math.sin(ang)
        return Node((nx, ny), parent=from_node)


    def compute_path( self, draw_graph: bool = False ):
        self.tree = [ Node( self.start ) ]
        for _ in range(self.max_iter):
            rnd = self.sample_point()
            nearest = self.nearest(rnd)
            new = self.steer(nearest, rnd)
            self.tree.append(new)
            # Verifica se alcançou o Goal
            if distance(new.position, self.goal) <= self.step_size:
                path = []
                cur = new
                while cur:
                    path.append(cur.position)
                    cur = cur.parent
                if draw_graph:
                    self._save_heatmap_with_path(
                        path,
                        resolution = 200,
                        filename = "heatmap_with_path.jpg")
                return list(reversed(path))
        return None


    def _reconstruct_path( self, node ):
        path, cur = [], node
        while cur:
            path.append(cur.position)
            cur = cur.parent
        return list(reversed(path))


    def _save_heatmap_with_path( self, path, resolution = 200, filename = "heatmap.jpg"):
        xs = np.linspace(-self.width/2,  self.width/2,  resolution)
        ys = np.linspace(-self.height/2, self.height/2, resolution)
        U  = np.empty((resolution, resolution), dtype=float)

        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                U[i, j] = self.compute_potential( x, y )

        fig, ax = plt.subplots(figsize=(6,5))
        cs = ax.contourf(
            xs, ys, U,
            levels = 50,
            cmap = 'viridis'
        )
        fig.colorbar(cs, ax=ax, label='U normalizado')

        if path:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            ax.plot(px, py, '-r', linewidth=2, label='RRT path')
            ax.scatter(px, py, c='red', s=20)
            ax.legend()

        ax.set_title('Heatmap de Potenciais + Caminho RRT')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.axis('equal')

        fig.savefig(filename, dpi=1200, format='jpg')
        plt.close(fig)