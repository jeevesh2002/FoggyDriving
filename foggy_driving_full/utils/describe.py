

def describe():
    print("""
FoggyDriving MDP Definition

S (State Space):
  Continuous vector in [0,1]^(4+number of lidar signals):
    - lane_onehot (2 dims)
    - normalized speed
    - normalized fog level
    - normalized lidar distances
  Full state: s = [lane_1hot, speed_norm, fog_norm, lidar_1...lidar_n]

A (Action Space):
  Discrete actions A = {0,1,2,3,4}:
    0 = maintain
    1 = accelerate
    2 = brake
    3 = lane left
    4 = lane right

p (Transition Dynamics):
  Stochastic transitions due to:
    - random fog changes
    - random obstacle car speeds and spawns
    - noisy lidar readings
  Ego dynamics:
    - lane shifts by at most ±1
    - speed increases/decreases within [v_min, v_max]
  Next state s' sampled from p(s'|s,a).

d0 (Initial State Distribution):
  Randomized environment reset:
    - random fog level
    - random number of cars (5–8)
    - random car distances and speeds
    - ego lane = 1, ego speed = mid-range
  Defines distribution d0(s) = P(s0 = s).

R (Reward Function):
  R(s,a,s') =
    speed'               (normal step)
    speed' - 50          (collision)
    speed' + 20          (surviving full episode)
  Encourages fast but safe driving.

γ (Discount Factor):
  gamma = 0.99
  Long-horizon control appropriate for driving.

""")
