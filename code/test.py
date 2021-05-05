# This script tests fully trained models

import sys
import pickle
from keras.models import load_model
from rlenvs import load_env

def test_model(env, model, folder, num_episodes=10):
    print("Beginning Testing...")
    total_reward = 0.0
    for episode in range(num_episodes):
        self.env.reset()
        obs = self.env.observation()
        for step in range(self.env.nsteps):
            action = self.get_action(state)
            observation, reward, done, info = self.env.step(action)
            if done: break
        total_reward += self.env.reward()
        self.env.render(title=f"Test Mission {episode}",
                        file=f"{folder}/runs/test{episode}.png")
    print("\nCompleted Testing!")
    print(f"Average Reward: {total_reward/num_episodes}")


m = int(input("(1) ConstantSimple2D\n(2) RandomSimple2D\n(3) ConstantComplex2D\n(4) RandomComplex2D\nSelect Test Model: "))

if (m == 1): path = "../saved/ConstantSimple2D"
elif (m == 2): path = "../saved/RandomSimple2D"
elif (m == 3): path = "../saved/ConstantComplex2D"
elif (m == 4): path = "../saved/RandomComplex2D"
else: print('invalid choice'); sys.exit()

env = load_mission(f"{folder}/env.pickle", m)
model = load_model(f"{folder}/model.h5")
test_model(env, model, folder)
