import gym
import numpy as np
from gym import spaces

from TSP_view_2D import TSPView2D


class TSPEasyEnv(gym.Env):

    def render(self, mode="human", close=False):

        if self.tsp_view is None:
            self.tsp_view = TSPView2D(self.n_tanks, self.map_quad, grid_size=25)

        return self.tsp_view.update(self.agt_at_depot, self.depot_x, self.depot_y, self.o_delivery,
                                    self.o_x, self.o_y, self.agt_x, self.agt_y, mode)

    def __init__(self, n_tanks=4, map_quad=(2, 2), max_time=50, 
                randomized_tanks=False):

        self.tsp_view = None
        self.map_quad = map_quad

        self.o_y = []
        self.o_x = []
        
        self.randomized_tanks = randomized_tanks

        self.n_tanks = n_tanks
        self.depot_x = 0
        self.depot_y = 0

        self.agt_x = None
        self.agt_y = None

        self.o_delivery = []
        self.o_time = []
        self.agt_at_depot = None
        self.agt_time = None

        self.max_time = max_time

        self.map_min_x = - map_quad[0]
        self.map_max_x = + map_quad[0]
        self.map_min_y = - map_quad[1]
        self.map_max_y = + map_quad[1]

        # agent x,
        agt_x_min = [self.map_min_x]
        agt_x_max = [self.map_max_x]
        # agent y,
        agt_y_min = [self.map_min_y]
        agt_y_max = [self.map_max_y]
        # n_tanks for x positions of tanks,
        o_x_min = [self.map_min_x for i in range(n_tanks)]
        o_x_max = [self.map_max_x for i in range(n_tanks)]
        # n_tanks for y positions of tanks,
        o_y_min = [self.map_min_y for i in range(n_tanks)]
        o_y_max = [self.map_max_y for i in range(n_tanks)]

        # whether delivered or not, 0 not delivered, 1 delivered
        o_delivery_min = [0] * n_tanks
        o_delivery_max = [1] * n_tanks

        # whether agent is at depot or not
        agt_at_depot_max = 1
        agt_at_depot_min = 0

        # Time since tanks have been placed
        o_time_min = [0] * n_tanks
        o_time_max = [max_time] * n_tanks

        # Time since start
        agt_time_min = 0
        agt_time_max = max_time

        self.observation_space = spaces.Box(
            low=np.array(
                agt_x_min + agt_y_min + o_x_min + o_y_min + [0] + [0] + o_delivery_min + [
                    agt_at_depot_min] + o_time_min + [
                    agt_time_min] + [0]),
            high=np.array(
                agt_x_max + agt_y_max + o_x_max + o_y_max + [0] + [0] + o_delivery_max + [
                    agt_at_depot_max] + o_time_max + [
                    agt_time_max] + [self.max_time]),
            dtype=np.int16
        )

        # Action space, UP, DOWN, LEFT, RIGHT
        self.action_space = spaces.Discrete(4)

    def reset(self):

        self.depot_x = 0
        self.depot_y = 0
        self.agt_x = self.depot_x
        self.agt_y = self.depot_y
        if self.randomized_tanks:
            # Enforce uniqueness of tanks, to prevent multiple tanks being placed on the same points
            # And ensure actual tanks in the episode are always == n_tanks as expected
            tanks=[]
            while len(sorted(set(tanks))) != self.n_tanks:
                tanks = [self.__receive_order() for i in range(self.n_tanks)]
        else:
            tanks = [(-2, -2), (1,1), (2,0), (0, -2)] 
        self.o_x = [x for x, y in tanks]
        self.o_y = [y for x, y in tanks]
        self.o_delivery = [0] * self.n_tanks
        self.o_time = [0] * self.n_tanks
        self.agt_at_depot = 1
        self.agt_time = 0

        return self.__compute_state()

    def step(self, action):

        done = False
        reward_before_action = self.__compute_reward()
        self.__play_action(action)
        reward = self.__compute_reward() - reward_before_action

        # If agent completed the route and returned back to start, give additional reward
        if (np.sum(self.o_delivery) == self.n_tanks) and self.agt_at_depot:
            done = True
            reward += self.max_time * 0.1

        # If there is timeout, no additional reward
        if self.agt_time >= self.max_time:
            done = True

        info = {}
        return self.__compute_state(), reward, done, info

    def __play_action(self, action):

        if action == 0:  # UP
            self.agt_y = min(self.map_max_y, self.agt_y + 1)
        elif action == 1:  # DOWN
            self.agt_y = max(self.map_min_y, self.agt_y - 1)
        elif action == 2:  # LEFT
            self.agt_x = max(self.map_min_x, self.agt_x - 1)
        elif action == 3:  # RIGHT
            self.agt_x = min(self.map_max_x, self.agt_x + 1)
        else:
            raise Exception("action: {action} is invalid")

        # Check for deliveries
        for ix in range(self.n_tanks):
            if self.o_delivery[ix] == 0:
                if (self.o_x[ix] == self.agt_x) and (self.o_y[ix] == self.agt_y):
                    self.o_delivery[ix] = 1

        # Update the time for the waiting tanks
        for ix in range(self.n_tanks):
            if self.o_delivery[ix] == 0:
                self.o_time[ix] += 1

        # Update time since agent left depot
        self.agt_time += 1

        # Check if agent is at depot
        self.agt_at_depot = int((self.agt_x == self.depot_x) and (self.agt_y == self.depot_y))

    def __compute_state(self):
        return [self.agt_x] + [self.agt_y] + self.o_x + self.o_y + [self.depot_x] + [
            self.depot_y] + self.o_delivery + [
                   self.agt_at_depot] + self.o_time + [
                   self.agt_time] + [(self.max_time - self.agt_time)]

    def __receive_order(self):

        # Generate a single order, not at the center (where the depot is)
        self.order_x = \
            np.random.choice([i for i in range(self.map_min_x, self.map_max_x + 1) if i != self.depot_x], 1)[0]
        self.order_y = \
            np.random.choice([i for i in range(self.map_min_y, self.map_max_y + 1) if i != self.depot_y], 1)[0]

        return self.order_x, self.order_y

    def __compute_reward(self):
        return np.sum(np.asarray(self.o_delivery) * self.max_time / (np.asarray(self.o_time) + 0.0001)) \
               - self.agt_time

class TSPMediumEnv(TSPEasyEnv):
    def __init__(self, n_tanks=8, map_quad=(3, 3), max_time=400, randomized_tanks=True):
        super().__init__(n_tanks, map_quad, max_time, randomized_tanks)

class TSPHardEnv(TSPEasyEnv):
    def __init__(self, n_tanks=10, map_quad=(10, 10), max_time=5000, randomized_tanks=True):
        super().__init__(n_tanks, map_quad, max_time, randomized_tanks)
