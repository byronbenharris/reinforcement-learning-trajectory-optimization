# This file trains a DQN model to solve a mission

import os
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pkl

from agents import DQN
from rlenvs import (
    ConstantSimple2DMissionEnv,
    RandomSimple2DMissionEnv,
    ConstantComplex2DMissionEnv,
    RandomComplex2DMissionEnv
)


class MissionAgent(DQN):

    """
    The MissionAgent is the network which provides the actions for each state.
    It inherits from agents.DQN which includes the core logic for a Deep-Q Network.
    MissionAgent implements specialized training and validation methods for SolarMissions.

    The Deep-Q Network needs the following:
        env -- the environment to train the network on
        lr -- the learning rate of the optimizer
        gamma -- indicates how much the model should look at history
        epsilon -- the randmoization factor for training
        epsilon_decay -- the decay of epsilon as training progresses, range: (0,1)

    `train` and `validate` are the outward facing methods of the class.
    """

    def __init__(self, env, lr, gamma, epsilon, epsilon_decay):
        super().__init__(env, lr, gamma, epsilon, epsilon_decay)
        self.step_count = []
        self.min_dist = []
        self.final_dist = []
        self.delta_v = []
        self.rewards = []
        self.rewards_avg = []

    def train(self, num_episodes, val_every, folder):

        """
        `train` will run training on the model for ::num_episodes:: adjusting the weights
        as dictated by the optimizer. Every ::val_every:: episodes will rendered as a plot
        within ::folder::.

        The model also tracks metrics as it goes. For each episode, it calculates
        the number of steps the episode took, minimum distance the rocket got to the target,
        thrust used, and reward.
        """

        print("\nBeginning Training...")
        for episode in range(num_episodes):

            self.env.reset()
            obs = self.env.observation()
            state = np.reshape(obs, [1, self.num_observation_space])
            done = False

            while not done:
                received_action = self.get_action(state)
                next_state, reward, done, info = self.env.step(received_action)
                next_state = np.reshape(next_state, [1, self.num_observation_space])
                self.add_to_replay_memory(state, received_action, reward, next_state, done)
                state = next_state
                self.update_counter()
                self.learn_and_update_weights_by_reply()

            if episode % val_every == 0:
                self.env.render(title=f"Train Mission {episode}",
                                file=f"{folder}/plots/train/{episode}.png")

            self.step_count.append(self.env.mission.step_count)
            self.min_dist.append(self.env.mission.min_dist)
            self.final_dist.append(self.env.mission.dist)
            self.delta_v.append(self.env.mission.rocket.dv_sum)
            self.rewards.append(self.env.reward())
            if len(self.rewards) >= 100:
                self.rewards_avg.append(np.mean(self.rewards[-100:]))

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            print(f"Episode {episode}" +
                  f"\tSteps: {self.env.mission.step_count}" +
                  f"\tFinal Dist: {self.env.mission.dist:.3f}" +
                  f"\tMin Dist: {self.env.mission.min_dist:.3f}" +
                  f"\tDelta V: {self.env.mission.rocket.dv_sum:.3f}" +
                  f"\tReward: {reward:.3f}" +
                  f"\tEpsilon: {self.epsilon:.3f}")

        print("Completed Training!")

    def validate(self, folder, num_episodes):

        """
        `validate` tests the model on ::num_episodes:: and stores the results to
        in ::folder::. This does NOT modify model weights.
        """

        print("\nBeginning Validation...")
        total_reward = 0.0

        for episode in range(num_episodes):

            self.env.reset()
            obs = self.env.observation()
            state = np.reshape(obs, [1, self.num_observation_space])
            done = False

            while not done:
                action = self.get_action(state, test=True)
                obs, reward, done, info = self.env.step(action)
                state = np.reshape(obs, [1, self.num_observation_space])

            total_reward += self.env.reward()

            self.env.render(title=f"Validate Mission {episode}",
                            file=f"{folder}/plots/validate/{episode}.png")

            print(f"Validate {episode}" +
                  f"\tSteps: {self.env.mission.step_count}" +
                  f"\tFinal Dist: {self.env.mission.dist:.3f}" +
                  f"\tMin Dist: {self.env.mission.min_dist:.3f}" +
                  f"\tDelta V: {self.env.mission.rocket.dv_sum:.3f}" +
                  f"\tReward: {self.env.reward():.3f}")

        print(f"Average Reward: {total_reward/num_episodes}")
        print("Completed Validation!")

    def plot_metrics(self, folder):
        self.plot_rewards(file=f"{folder}/metrics/training_reward.png")
        self.plot_dists(file=f"{folder}/metrics/training_dist.png")
        self.plot_deltav(file=f"{folder}/metrics/training_deltav.png")

    def plot_rewards(self, file=''):
        plt.figure(0); plt.clf()
        plt.plot(self.rewards)
        plt.plot(self.rewards_avg)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title("Training Rewards")
        plt.axis('tight')
        plt.grid(True)
        if file: plt.savefig(file)
        else: plt.show()

    def plot_dists(self, file=''):
        plt.figure(0); plt.clf()
        plt.plot(self.min_dist)
        plt.plot(self.final_dist)
        plt.xlabel('Episode')
        plt.ylabel('Distance (AU)')
        plt.title("Training Distances")
        plt.axis('tight')
        plt.grid(True)
        if file: plt.savefig(file)
        else: plt.show()

    def plot_deltav(self, file=''):
        plt.figure(0); plt.clf()
        plt.plot(self.delta_v)
        plt.xlabel('Episode')
        plt.ylabel('Delta V')
        plt.title("Training Velocities")
        plt.axis('tight')
        plt.grid(True)
        if file: plt.savefig(file)
        else: plt.show()

    def save_metrics(self, folder):
        # save metrics to pickles for long-term storage
        pkl.dump(self.step_count, open(f"{folder}/metrics/steps.pkl", "wb"))
        pkl.dump(self.min_dist, open(f"{folder}/metrics/min_dists.pkl", "wb"))
        pkl.dump(self.final_dist, open(f"{folder}/metrics/final_dists.pkl", "wb"))
        pkl.dump(self.delta_v, open(f"{folder}/metrics/deltavs.pkl", "wb"))
        pkl.dump(self.rewards, open(f"{folder}/metrics/rewards.pkl", "wb"))
        pkl.dump(self.rewards_avg, open(f"{folder}/metrics/rewards_avg.pkl", "wb"))


### HYPERPARAMETERS ###

TAU = 0.001
NSTEPS = 5000
NPLANETS = 2
NEPISODES = 1500
PLOT_EVERY = 1
NVALIDATE = 25

epsilon = 1.0
gamma = 0.5

### MAIN ###

m = int(input("(1) ConstantSimple2D\n" +
              "(2) RandomSimple2D\n" +
              "(3) ConstantComplex2D\n" +
              "(4) RandomComplex2D\n" +
              "Select Test Model: "))

if (m == 1):
    folder = "../saved/ConstantSimple2D"
    env = ConstantSimple2DMissionEnv(TAU, NSTEPS)
    epsilon_decay = 0.99; lr = 0.01
elif (m == 2):
    folder = "../saved/RandomSimple2D"
    env = RandomSimple2DMissionEnv(TAU, NSTEPS)
    epsilon_decay = 0.997; lr = 0.001
elif (m == 3):
    folder = "../saved/ConstantComplex2D"
    env = ConstantComplex2DMissionEnv(TAU, NPLANETS, NSTEPS)
    epsilon_decay = 0.99; lr = 0.01
elif (m == 4):
    folder = "../saved/RandomComplex2D"
    env = RandomComplex2DMissionEnv(TAU, NPLANETS, NSTEPS)
    epsilon_decay = 0.997; lr = 0.001
else:
    print('invalid choice')
    sys.exit()

# lr = float(input("Set Learning Rate: "))
c = input("Custom Run Marker: ")
folder += f"_lr={lr}_m=64_{c}"

# make sure all important directories exist
os.makedirs(f"{folder}/", exist_ok=True)
os.makedirs(f"{folder}/metrics/", exist_ok=True)
os.makedirs(f"{folder}/plots/", exist_ok=True)
os.makedirs(f"{folder}/plots/train/", exist_ok=True)
os.makedirs(f"{folder}/plots/test/", exist_ok=True)
os.makedirs(f"{folder}/plots/validate/", exist_ok=True)

# save the env in a pickle
pkl.dump(env, open(f"{folder}/env.pkl", "wb"))
# create and train a model
# if you KeyboardInterrupt when ready to quit training
# model will still be saved
model = MissionAgent(env, lr, gamma, epsilon, epsilon_decay)
try: model.train(NEPISODES, PLOT_EVERY, folder)
except KeyboardInterrupt: pass
# save all important model info
model.save(f"{folder}/model.h5")
model.plot_metrics(folder)
model.save_metrics(folder)
# see how well the model did
model.validate(folder, NVALIDATE)
