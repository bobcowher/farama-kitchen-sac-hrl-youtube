import gymnasium as gym
import numpy as np
from gym_robotics_custom import RoboGymObservationWrapper
from buffer import ReplayBuffer
from controller import Controller
import pygame
import time

if __name__ == '__main__':
    env_name="FrankaKitchen-v1"
    max_episode_steps=500
    replay_buffer_size = 1000000


    task = 'microwave'
    task_no_spaces = task.replace(" ", "_")
    
    env = gym.make(env_name, max_episode_steps=max_episode_steps, tasks_to_complete=[task],
                   render_mode='human', autoreset=False)
    
    env = RoboGymObservationWrapper(env, goal=task)
    
    print(env.env.env.env.env.model.opt.gravity)

    state, _ = env.reset()

    state_size = state.shape[0]

    memory = ReplayBuffer(replay_buffer_size, input_size=state_size, n_actions=env.action_space.shape[0])

    memory.load_from_csv(filename=f"checkpoints/human_memory_{task_no_spaces}.npz")

    starting_memory_size = memory.mem_ctr

    print(f"Starting memory size is {starting_memory_size}")

    controller = Controller()

    while True:
        episode_steps = 0
        done = False
        state, info = env.reset()

        while not done and episode_steps < max_episode_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_h and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        env.render()
                    action = controller.get_action()
            
            action = controller.get_action()

            if(action is not None):
                next_state, reward, done, _, _ = env.step(action)
                mask = 1 if episode_steps == max_episode_steps else float(not done)
                memory.store_transition(state, action, reward, next_state, mask)
                print(f"Episode step: {episode_steps} Reward: {reward} Successfully added \
                       {memory.mem_ctr - starting_memory_size} steps to memory. Total: {memory.mem_ctr}")
                state = next_state
                episode_steps += 1
            time.sleep(0.05)

        memory.save_to_csv(filename=f"checkpoints/human_memory_{task_no_spaces}.npz")
    
    env.close()


    