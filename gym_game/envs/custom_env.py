import gym
from gym import spaces
import numpy as np
from gym_game.envs.pygame_2d import PyGame2D

# klasa odpowiedzialna za integracje gry z biblioteką opengymai
# odwołujemy się do niej w main.py
class CustomEnv(gym.Env):

    # inicjalizacja 
    def __init__(self):
        self.pygame = PyGame2D()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.array([0, 0, 0, 0, 0]), np.array([10, 10, 10, 10, 10]), dtype=np.int)

    # resetujemy gre po każdej próbie
    def reset(self):
        del self.pygame
        self.pygame = PyGame2D()
        obs = self.pygame.observe()
        return obs

    # zwraca wyniki state'a
    def step(self, action):
        self.pygame.action(action)
        obs = self.pygame.observe()
        reward = self.pygame.evaluate()
        done = self.pygame.is_done()
        return obs, reward, done, {}

    # render
    def render(self, mode="human", close=False):
        self.pygame.view()

    # skróty klawiszowe (s, r, t, y, u)
    def handle_events(self, table):
        self.pygame.handle_events(table)
