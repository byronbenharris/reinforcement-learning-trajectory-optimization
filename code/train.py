# This file trains a DQN model to solve a mission

import os
import pickle
import numpy as np
import matplotlib as plt

from agents import DQN
from rlenvs import (
    ConstantSimple2DMissionEnv,
    RandomSimple2DMissionEnv,
    ConstantComplex2DMissionEnv,
    RandomComplex2DMissionEnv
)


class MissionAgent(DQN):

    """
    """

    def __init__(self, env, lr, gamma, epsilon, epsilon_decay):
        super().__init__(env, lr, gamma, epsilon, epsilon_decay)
        self.training_rewards = []
        self.training_rewards_avg = []

    def train(self, num_episodes, val_every, folder):

        print("Beginning Training...")
        for episode in range(num_episodes):

            self.env.reset()
            obs = self.env.observation()
            state = np.reshape(obs, [1, self.num_observation_space])

            for step in range(self.env.nsteps):
                received_action = self.get_action(state)
                next_state, reward, done, info = self.env.step(received_action)
                next_state = np.reshape(next_state, [1, self.num_observation_space])
                self.add_to_replay_memory(state, received_action, reward, next_state, done)
                state = next_state
                self.update_counter()
                self.learn_and_update_weights_by_reply()
                if done: break

            if episode % val_every == 0:
                self.env.render(title=f"Train Mission {episode}",
                                file=f"{folder}/runs/train{episode}.png")

            reward = self.env.reward()
            self.training_rewards.append(reward)
            last_hundred = self.training_rewards
            if len(self.training_rewards) > 100:
                last_hundred = self.training_rewards[-100:]
            self.training_rewards_avg.append(sum(last_hundred) / len(last_hundred))

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            print(f"Episode {episode}\tSteps: {self.env.mission.step_count}\tMin Dist: {self.mission.min_dist}\tDelta V: {self.mission.rocket.total_dv}\tReward: {reward}\tEpsilon: {self.epsilon}")

        print("Completed Training!")

    def validate(self, folder, num_episodes=10):
        print("Beginning Validation...")
        total_reward = 0.0
        for episode in range(num_episodes):
            self.env.reset()
            obs = self.env.observation()
            for step in range(self.env.nsteps):
                action = self.get_action(state)
                observation, reward, done, info = self.env.step(action)
                if done: break
            total_reward += self.env.reward()
            self.env.render(title=f"Validate Mission {episode}",
                            file=f"{folder}/runs/validate{episode}.png")
        print("\nCompleted Validation!")
        print(f"Average Reward: {total_reward/num_episodes}")

    def plot_rewards(self, file=''):
        plt.figure(0); plt.clf()
        plt.plot(self.source.xplot,self.source.yplot,'b-')
        plt.plot(self.source.r[0],self.source.r[1],'o-')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title("Training Rewards")
        plt.axis('tight')
        plt.axis('equal')
        plt.grid(True)
        if file: plt.savefig(file)
        else: plt.show()


m = int(input("(1) ConstantSimple2D\n(2) RandomSimple2D\n(3) ConstantComplex2D\n(4) RandomComplex2D\nSelect Test Model: "))

if (m == 1):
    folder = "../saved/ConstantSimple2D"
    env = ConstantSimple2DMissionEnv()
elif (m == 2):
    folder = "../saved/RandomSimple2D"
    env = RandomSimple2DMissionEnv()
elif (m == 3):
    folder = "../saved/ConstantComplex2D"
    env = ConstantComplex2DMissionEnv()
elif (m == 4):
    folder = "../saved/RandomComplex2D"
    env = RandomComplex2DMissionEnv()
else:
    print('invalid choice')
    sys.exit()

pickle.dump(env.mission.state, open(f"{folder}/env.pkl", "wb"))
lr = 0.001; epsilon = 1.0; epsilon_decay = 0.98; gamma = 0.99
model = MissionAgent(env, lr, gamma, epsilon, epsilon_decay)
model.train(100, 1, folder)
model.save(f"{folder}/model.h5")
model.plot_rewards(file=f"{folder}/training.pdf")
pickle.dump(model.rewards, open(f"{folder}/rewards.pkl", "wb"))
pickle.dump(model.rewards_avg, open(f"{folder}/rewards_avg.pkl", "wb"))
self.validate(folder)
