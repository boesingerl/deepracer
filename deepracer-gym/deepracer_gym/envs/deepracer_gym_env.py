import numpy as np
import gym
from deepracer_gym.zmq_client import DeepracerEnvHelper

class DeepracerGymEnv(gym.Env):
    def __init__(self, host="127.0.0.1", port=8888):
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Dict({'STEREO_CAMERAS': gym.spaces.Box(low=0,high=255,shape=(120,160,2), dtype='uint8')})
        self.deepracer_helper = DeepracerEnvHelper(host=host, port=port)
    
    def reset(self):
        observation = self.deepracer_helper.env_reset()
        return observation
    
    def step(self, action):
        rl_coach_obs = self.deepracer_helper.send_act_rcv_obs(action)
        observation, reward, done, info = self.deepracer_helper.unpack_rl_coach_obs(rl_coach_obs)
        return observation, reward, done, info

if __name__ == '__main__':
    env = DeepracerGymEnv(host="127.0.0.1", port=8888)
    obs = env.reset()
    steps_completed = 0
    episodes_completed = 0
    total_reward = 0
    for _ in range(500):
        observation, reward, done, info = env.step(np.random.randint(5))
        steps_completed += 1 
        total_reward += reward
        if done:
            episodes_completed += 1
            print("Episodes Completed:", episodes_completed, "Steps:", steps_completed, "Reward", total_reward)
            steps_completed = 0
            total_reward = 0
