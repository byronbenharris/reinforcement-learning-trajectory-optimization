# This file implements the environments for RL using the solar classes

import gym
from gym import spaces

import pickle
import numpy as np

from solar import (
    CreateRandomSimpleHighThrustMission,
    CreateRandomComplexHighThrustMission,
)


class MissionEnv(gym.Env):

    def __init__(self, mission, nplanets, nsteps):
        self.nsteps = nsteps
        self.nplanets = nplanets
        self.mission = mission
        self.action_space = spaces.Box(-7.5, +7.5, (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf,
            shape=(15+(5*nplanets),), dtype=np.float32)

    def observation(self):
        return self.mission.observation()

    def reward(self):
        return self.mission.reward()

    def render(self, title, file):
        self.mission.plot_mission(title=title, file=file)

    def step(self, action):
        self.mission.step(action)
        return self.mission.observation(), self.mission.reward(), self.mission.done(), {}


class ConstantSimple2DMissionEnv(MissionEnv):

    def __init__(self, nsteps=500):
        mission = CreateRandomSimpleHighThrustMission(0.001)
        super().__init__(mission, 0, nsteps)

    def reset(self):
        self.mission.reset()


class RandomSimple2DMissionEnv(MissionEnv):

    def __init__(self, nsteps=500):
        mission = CreateRandomSimpleHighThrustMission(0.001)
        super().__init__(mission, 0, nsteps)

    def reset(self):
        self.mission = CreateRandomSimpleHighThrustMission(0.001)


class ConstantComplex2DMissionEnv(MissionEnv):

    def __init__(self, nplanets, nsteps=500):
        mission = CreateRandomComplexHighThrustMission(0.001, nplanets)
        super().__init__(mission, nplanets, nsteps)

    def reset(self):
        self.mission.reset()


class RandomComplex2DMissionEnv(MissionEnv):

    def __init__(self, nplanets, nsteps=500):
        mission = CreateRandomComplexHighThrustMission(0.001, nplanets)
        super().__init__(mission, nplanets, nsteps)

    def reset(self):
        self.mission = CreateRandomComplexHighThrustMission(0.001, self.nplanets)


def load_mission(file, type):
    if type == 2: return CreateRandomSimpleHighThrustMission()
    if type == 4: pass
