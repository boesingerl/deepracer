import gym
import numpy as np

import cv2

import deepracer_gym


def render(obs):
    cv2.imshow('left', obs['STEREO_CAMERAS'][:,:,0])
    cv2.imshow('right', obs['STEREO_CAMERAS'][:,:,1])
    cv2.waitKey(1)
    
    
env = gym.make('deepracer_gym:deepracer-v0')

obs = env.reset()

print("Deepracer Environment Connected succesfully")

steps_completed = 0
episodes_completed = 0
total_reward = 0

for _ in range(500):
    observation, reward, done, info = env.step(np.random.randint(5))
  
    steps_completed += 1 
    total_reward += reward
  
    render(observation)
    
    if done:
        episodes_completed += 1
        print("Episodes Completed:", episodes_completed, "Steps:", steps_completed, "Reward", total_reward)
        steps_completed = 0
        total_reward = 0
        


