import torch.nn as nn

from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, SubprocVecEnv, VecEnvWrapper, VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env import VecExtractDictObs
from stable_baselines3.common.monitor import Monitor

class BasicExtractor(BaseFeaturesExtractor):
    """
    Basic CNN Feature extractor with two layers, 16 and 32 channels
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param num_envs: (int) Number of dummy environments in which this feature extractor is used
    """

    def __init__(self, observation_space, features_dim = 128, num_envs=1, **kwargs):
        super(BasicExtractor, self).__init__(observation_space, features_dim)
        
        self.linear = nn.Sequential(
            Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(16),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(32),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512,128)
        ).to('cuda:0').float()
        self.num_envs = num_envs
        
    def forward(self, observations,  **kwargs):
        inp = tensor_to_features(observations[:,1:,:,:]).float()
        return self.linear(inp)
    
    
policy_kwargs = dict(
    features_extractor_class=BasicExtractor,
    features_extractor_kwargs=dict(features_dim=128, num_envs=1),
)


def linear_schedule(initial_value: float) :
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func
