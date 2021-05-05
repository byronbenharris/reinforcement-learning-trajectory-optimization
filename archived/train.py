# This script trains and saves RL models

import os

from rlenvs import (
    ConstantSimple2DMissionEnv,
    RandomSimple2DMissionEnv,
    ConstantComplex2DMissionEnv,
    RandomComplex2DMissionEnv
)

from test import compute_avg_return

import tensorflow as tf
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import categorical_q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

tf.compat.v1.enable_v2_behavior()


num_iterations = 15000
log_interval = 200
eval_interval = 1000

initial_collect_steps = 1000
collect_steps_per_iteration = 1
replay_buffer_capacity = 100000

gym_env = ConstantSimple2DMissionEnv()
env = tf_py_environment.TFPyEnvironment(gym_env)

categorical_q_net = categorical_q_network.CategoricalQNetwork(
    env.observation_spec(),
    env.action_spec(),
    num_atoms=51,
    fc_layer_params=(100,)
)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train_step_counter = tf.compat.v2.Variable(0)

agent = categorical_dqn_agent.CategoricalDqnAgent(
    env.time_step_spec(), env.action_spec(),
    categorical_q_network = categorical_q_net, optimizer = optimizer,
    min_q_value = -20, max_q_value = 20, n_step_update = 2,
    td_errors_loss_fn = common.element_wise_squared_loss,
    gamma = 0.99, train_step_counter = train_step_counter
)

agent.initialize()

random_policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(), env.action_spec())
compute_avg_return(env, random_policy, 10)

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

# Please also see the metrics module for standard implementations of different
# metrics.

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec, batch_size=env.batch_size, max_length=100000
)

def collect_step(environment, policy):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)

for _ in range(initial_collect_steps):
    collect_step(env, random_policy)

# This loop is so common in RL, that we provide standard implementations of
# these. For more details see the drivers module.

# Dataset generates trajectories with shape [BxTx...] where T = n_step_update + 1.
dataset = replay_buffer.as_dataset(num_parallel_calls=3,
    sample_batch_size=64, num_steps=n_step_update + 1).prefetch(3)

iterator = iter(dataset)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(env, agent.policy)
returns = [avg_return]

for _ in range(num_iterations):

    # Collect a few steps using collect_policy and save to the replay buffer.
    for _ in range(collect_steps_per_iteration):
        collect_step(env, agent.collect_policy)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience)

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(env, agent.policy)
        print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
        returns.append(avg_return)

steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim(top=550)
