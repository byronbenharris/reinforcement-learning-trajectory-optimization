# This script tests fully trained models

import sys
import _pickle as pkl
import numpy as np
from keras.models import load_model

def test_model(env, model, folder, num_episodes=10):

    """
    tests the model on ::num_episodes:: and reports the results.
    Does not modify weights.
    """

    print("\nBeginning Testing...")
    total_reward = 0.0
    obs_size = env.observation_space.shape[0]

    for episode in range(num_episodes):

        env.reset()
        obs = env.observation()
        state = np.reshape(obs, [1, obs_size])
        done = False

        while not done:
            action = np.argmax(model.predict(state)[0])
            obs, reward, done, info = env.step(action)
            state = np.reshape(obs, [1, obs_size])

        total_reward += env.reward()

        env.render(title=f"Test Mission {episode}",
                   file=f"{folder}/plots/test/{episode}.png")

        print(f"Test {episode}" +
              f"\tSteps: {env.mission.step_count}" +
              f"\tMin Dist: {env.mission.min_dist}" +
              f"\tDelta V: {env.mission.rocket.total_dv}" +
              f"\tReward: {env.reward()}")

    print(f"Average Reward: {total_reward/num_episodes}")
    print("Completed Testing!")


### MAIN ###

m = int(input("(1) ConstantSimple2D\n" +
              "(2) RandomSimple2D\n" +
              "(3) ConstantComplex2D\n" +
              "(4) RandomComplex2D\n" +
              "Select Test Model: "))

if (m == 1): path = "../saved/ConstantSimple2D"
elif (m == 2): path = "../saved/RandomSimple2D"
elif (m == 3): path = "../saved/ConstantComplex2D"
elif (m == 4): path = "../saved/RandomComplex2D"
else: print('invalid choice'); sys.exit()

# loads the environment and model
env = pkl.load(open(f"{path}/env.pkl", "rb"))
model = load_model(f"{path}/model.h5")
# runs a test on the model
test_model(env, model, path)
