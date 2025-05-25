from rSoccer.rsoccer_gym.Entities.Robot import Robot 
import numpy as np 


def print_matrix(name: str, M: np.ndarray) -> None:
    """
    Imprime a matriz M com formatação de 4 casas decimais.
    """
    print(f"\n{name}:")
    for row in M:
        print("[ " + "  ".join(f"{v:6.2f}" for v in row) + " ]")



if __name__ == "__main__":
      '''
      Teste da aplicação do campo potencial para o Robo 
      '''

      # 1. Instancia um robô e define a posição e velocidade dele 
      pos_robot = ( 5.0, 5.0, 0.0 ) # Posição no 1° quadrante 
      vel_robot = ( -1.0, -1.0 )    # Velocidade indo para a origem

      robot = Robot( id = 1, yellow = False )
      robot.update_pose( *pos_robot, theta = 0.0 ) 
      robot.update_velocity( *vel_robot, v_theta = 0.0 )
      
      A: float = 1.0         # Amplitude 
      sigma: float = 2.0     # Raio de influencia 
      beta: float = 2.0      # Peso da velocidade 
      robot.force_field.update_parameters( A, sigma, beta )
      # Pontos de teste para calcular o potencial
      '''
            Y
            ▲
            |                      
      3.00  O      O      O      O      
      2.50  |                           
      2.00  O      O      R      O      
      1.50  |                          
      1.00  O      O      O      O       
      0.50  |                           
      0.00  O------O------O------O------►
            0      1      2      3      X
      '''
      n_dim = 10
      xs = [ x for x in range(n_dim) ]
      ys = [ y for y in range(n_dim, 0, -1 ) ] 

      U  = np.zeros( (n_dim, n_dim), dtype = float )
      Fx = np.zeros( (n_dim, n_dim), dtype = float )
      Fy = np.zeros( (n_dim, n_dim), dtype = float )

      for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                  potential = robot.potential_at(x, y)
                  U[i, j]  = potential if potential > 0.0 else 0.0 
                  force_x, force_y = robot.force_at(x, y)
                  Fx[i, j] = force_x if force_x < 100 else 100 if force_x > 0.0 else 0.0 
                  Fy[i, j] = force_y if force_y < 100 else 100 if force_y > 0.0 else 0.0 

      print(f"Robot @ ({robot.x:.1f},{robot.y:.1f})  v=({robot.v_x:.1f},{robot.v_y:.1f})\n")
      print_matrix("Potencial U", U)
      print_matrix("Força Fx",   Fx)
      print_matrix("Força Fy",   Fy)