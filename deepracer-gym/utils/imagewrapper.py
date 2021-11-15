import sys
sys.path += ['..']

import gym
import cv2
import numpy as np

from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from typing import Dict, List, Optional, Tuple, Union
from skimage.transform import resize

from utils.image import get_otsu, get_disparity

class ImageWrap(gym.Wrapper):
    """
    An Image wrapper for our custom deepracer environment
    Applies otsu thresholding, and can apply disparity maps (disabled by default)
    Resizes observations into given size
    """


    def __init__(
        self,
        env: gym.Env,
        size: Tuple[int,int] = (64,64),
        disparity: bool = False
    ):
        super(ImageWrap, self).__init__(env=env)
        self.size = size
        self.disparity = disparity
        self.observation_space = gym.spaces.Box(low=0,high=255,shape=size+(5 if disparity else 4,), dtype='uint8')

    def transform_img(self,obs,disparity=False):
        """Transforms image by stacking otsu thresholding and disparity (if enabled) to itself"""
        # resize and get individual
        obs = cv2.resize(obs, dsize=self.size, interpolation=cv2.INTER_CUBIC)
        left, right = obs[:,:,0], obs[:,:,1]
        
        # compute otsu thresholding
        left_t = get_otsu(left)
        right_t = get_otsu(right)

        imgs = (left, right, left_t, right_t)
        
        if disparity:
            # compute disparity
            disparity = get_disparity(left,right)
            imgs += (disparity,)
        
        return np.stack(imgs,axis=-1)
    
    def reset(self, **kwargs) -> GymObs:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        obs = self.env.reset(**kwargs)
        obs = obs['STEREO_CAMERAS']
        
        return self.transform_img(obs, disparity=self.disparity)


    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, done, information
        """
        observation, reward, done, info = self.env.step(action)
        obs = self.transform_img(observation['STEREO_CAMERAS'], disparity=self.disparity)
        
        return obs, reward, done, info