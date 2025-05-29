from rSoccer.rsoccer_gym.Entities.PotentialField import PotentialField
from dataclasses import dataclass, field 

@dataclass()
class Ball:
    x: float = None
    y: float = None
    z: float = None
    theta: float = 0.0 
    
    v_x: float = 0.0
    v_y: float = 0.0
    v_z: float = 0.0
    v_theta: float = 0.0 

    # Campo potencial de força relativa à bola 
    potencial_field: PotentialField = field( init = False )

    def __post_init__(self):
        self.potencial_field = PotentialField( self  )
