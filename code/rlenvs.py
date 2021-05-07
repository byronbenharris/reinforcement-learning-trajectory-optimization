# This file implements the environments for RL using the solar classes

import gym
from gym import spaces

import numpy as np
import _pickle as pkl

from solar import (
    CreateRandomSimpleHighThrustMission,
    CreateRandomComplexHighThrustMission,
)


class MissionEnv(gym.Env):

    def __init__(self, mission, nplanets, nsteps):
        self.episode = 0
        self.nsteps = nsteps
        self.nplanets = nplanets
        self.mission = mission
        # self.action_space = spaces.Discrete(17)
        # self.action_space = spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(15+(5*nplanets),), dtype=np.float32)

    def observation(self):
        return self.mission.observation()

    def reward(self):
        return self.mission.reward()

    def render(self, title, file):
        self.mission.plot_mission(title=title, file=file)

    def step(self, action):
        self.mission.step(action)
        done = (self.mission.step_count > self.nsteps) or self.mission.done()
        return self.mission.observation(), self.mission.reward(), done, {}


class ConstantSimple2DMissionEnv(MissionEnv):

    def __init__(self, tau, nsteps):
        mission = CreateRandomSimpleHighThrustMission(tau)
        super().__init__(mission, 0, nsteps)

    def reset(self):
        self.mission.reset()


class RandomSimple2DMissionEnv(MissionEnv):

    def __init__(self, tau, nsteps):
        mission = CreateRandomSimpleHighThrustMission(tau)
        super().__init__(mission, 0, nsteps)

    def reset(self):
        self.episode += 1
        # only changes onces every 5 episodes to give model time to learn
        if (self.episode % 5 != 0): self.mission.reset()
        else: self.mission = CreateRandomSimpleHighThrustMission(self.mission.tau)



class ConstantComplex2DMissionEnv(MissionEnv):

    def __init__(self, tau, nplanets, nsteps):
        mission = CreateRandomComplexHighThrustMission(tau, nplanets)
        super().__init__(mission, nplanets, nsteps)

    def reset(self):
        self.mission.reset()


class RandomComplex2DMissionEnv(MissionEnv):

    def __init__(self, tau, nplanets, nsteps):
        mission = CreateRandomComplexHighThrustMission(tau, nplanets)
        super().__init__(mission, nplanets, nsteps)

    def reset(self):
        self.episode += 1
        # only changes onces every 5 episodes to give model time to learn
        if (self.episode % 5 != 0): self.mission.reset()
        else: self.mission = CreateRandomComplexHighThrustMission(self.mission.tau, self.nplanets)
