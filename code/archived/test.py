# This script tests fully trained models

import os

from rlenvs import (
    ConstantSimple2DMissionEnv,
    RandomSimple2DMissionEnv,
    ConstantComplex2DMissionEnv,
    RandomComplex2DMissionEnv
)

# plot_path = os.path.join(os.getcwd(), "../plots")
#
# env = ConstantSimple2DSolarEnv()
# for episode in range(20):
#     observation = env.reset()
#     for step in range(10000):
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             break
#     env.render(title=f"Mission {episode}", file=plot_path+f"/mission{episode}.png")
# env.close()
