import time
import gymnasium as gym
import numpy as np
from gym_robotics_custom import RoboGymObservationWrapper
from buffer import ReplayBuffer
from model import *
from agent import *
from meta_agent import *

if __name__ == '__main__':

    env_name="FrankaKitchen-v1"
    max_episode_steps=1500
    replay_buffer_size = 1000000
    tasks = ['top burner', 'microwave', 'hinge cabinet']
    gamma = 0.99
    tau = 0.005
    alpha = 0.1
    target_update_interval = 1
    updates_per_step = 4
    hidden_size = 512
    learning_rate = 0.00005
    batch_size = 64
    episodes = 3000


    env = gym.make(env_name, max_episode_steps=max_episode_steps, tasks_to_complete=tasks)
    env = RoboGymObservationWrapper(env)

    observation, info = env.reset()

    observation_size = observation.shape[0]

    meta_agent = MetaAgent(env, tasks, max_episode_steps=max_episode_steps)

    meta_agent.initialize_memory(augment_data=True, augment_rewards=True, augment_noise_ratio=0.1)

    meta_agent.initialize_agents()

    meta_agent.train(episodes=episodes, summary_writer_name=f'meta_agent')

    meta_agent.save_models()

    env.close()



    

    