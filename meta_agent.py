import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from model import *
from torch.utils.tensorboard import SummaryWriter
import datetime
from buffer import ReplayBuffer
import time
from gym_robotics_custom import RoboGymObservationWrapper
from agent import Agent
import random


class MetaAgent(object):
    
    def __init__(self, env, goal_list=['microwave'], replay_buffer_size=1000000, max_episode_steps=500):
        self.agent_dict = {}
        self.mem_dict: dict[str, ReplayBuffer] = {}
        goal_list_no_spaces = [a.replace(" ", "_") for a in goal_list]
        self.goal_dict: dict[str, str] = dict(zip(goal_list_no_spaces, goal_list))
        self.env = env
        self.agent: Agent = None
        self.replay_buffer_size = replay_buffer_size
        self.max_episode_steps = max_episode_steps

    def initialize_memory(self, augment_rewards=True, augment_data=True, augment_noise_ratio=0.1):

        for goal in self.goal_dict:
            self.env.set_goal(self.goal_dict[goal])
            observation, info = self.env.reset()
            observation_size = observation.shape[0]

            memory = ReplayBuffer(self.replay_buffer_size, input_size=observation_size,
                                  n_actions=self.env.action_space.shape[0], augment_rewards=augment_rewards,
                                  augment_data=augment_data, expert_data_ratio=0, augment_noise_ratio=augment_noise_ratio)
            
            self.mem_dict[goal] = memory
        
    def load_memory(self):
        for buffer in self.mem_dict:
            self.mem_dict[buffer].load_from_csv(filename=f"checkpoints/human_memory_{buffer}.npz")


    def initialize_agents(self, gamma=0.99, tau=0.005, alpha=0.1,
                          target_update_interval=2, hidden_size=512,
                          learning_rate=0.0001):
        
        for goal in self.goal_dict:
            self.env.set_goal(self.goal_dict[goal])
            observation, info = self.env.reset()
            observation_size = observation.shape[0]

            agent = Agent(observation_size, self.env.action_space, gamma=gamma, tau=tau, alpha=alpha,
                          target_update_interval=target_update_interval, hidden_size=hidden_size,
                          learning_rate=learning_rate, goal=goal)
            
            print(f"Loading checkpoint for {goal}")

            agent.load_checkpoint(evaluate=True)

            self.agent_dict[goal] = agent

    def save_models(self):
        for agent in self.agent_dict:
            self.agent_dict[agent].save_checkpoint()

    
    def test(self):
        action = None
        episode_reward = 0

        for goal in self.goal_dict:
            print(f"Attempting goal {goal}...")
            self.env.set_goal(self.goal_dict[goal])
            self.agent = self.agent_dict[goal]

            action, reward = self.agent.test(env=self.env, episodes=1, max_episode_steps=self.max_episode_steps, prev_action=action)

            episode_reward += reward
        
        return episode_reward


    def train(self, episodes, batch_size=64, summary_writer_name='meta_agent'):
        
        # Tensorboard
        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_' + summary_writer_name
        writer = SummaryWriter(summary_writer_name)

        updates = 0

        for episode in range(episodes):

            last_action = None
            action = None

            episode_reward = 0
            episode_steps = 0

            print(f"Starting episode: {episode}")

            num_samples = random.choice([1, 2])

            for goal in random.sample(list(self.goal_dict.keys()), num_samples):

                done = False
                self.env.set_goal(self.goal_dict[goal])
                state, _ = self.env.reset()

                while not done and episode_steps < self.max_episode_steps:

                    if last_action is not None:
                        action = last_action
                        last_action = None
                    else:
                        action = self.agent_dict[goal].select_action(state)
                    
                    if self.mem_dict[goal].can_sample(batch_size=64):
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.agent_dict[goal].update_parameters(self.mem_dict[goal],
                                                                                                                             batch_size,
                                                                                                                             updates)
                        writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                        writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                        writer.add_scalar('loss/policy', policy_loss, updates)
                        writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                        updates += 1
                    
                    next_state, reward, done, _, _ = self.env.step(action)

                    if reward == 1:
                        done = True
                        last_action = action
                    
                    episode_steps += 1
                    episode_reward += reward

                    # Ignore the done signal if it comes from hitting the time horizon.
                    mask = 1 if episode_steps == self.max_episode_steps else float(not done)

                    self.mem_dict[goal].store_transition(state, action, reward, next_state, mask)

                    state = next_state

                episode_reward = episode_reward / num_samples

                writer.add_scalar('reward/train', episode_reward, episode)
                writer.add_scalar('reward/episode_steps', episode_steps, episode)
                print("Episode: {}, Episode steps: {}, reward: {}".format(episode, episode_steps, episode_reward))

                if episode % 10 == 0:
                    self.save_models()


    

        