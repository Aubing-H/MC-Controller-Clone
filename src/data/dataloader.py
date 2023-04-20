import numpy as np
import os, math
import random
import lmdb
import json
import pickle
import torch
from torch.utils.data import Dataset
from typing import *
from tqdm import tqdm

from src.utils.utils import VideoHolder, ImageHolder
from torch.distributions.categorical import Categorical


def discrete_horizon(horizon):
    '''
    0 - 9: 0
    10 - 19: 1
    20 - 30: 2
    30 - 40: 3
    40 - 50: 4
    50 - 60: 5
    60 - 70: 6
    70 - 80: 7
    80 - 90: 8
    90 - 100: 9
    100 - 120: 10
    120 - 140: 11
    140 - 160: 12
    160 - 180: 13
    180 - 200: 14
    200 - ...: 15
    '''
    # horizon_list = [0]*25 + [1]*25 + [2]*25 + [3]*25 +[4]* 50 + [5]*50 + [6] * 700
    horizon_list = []
    for i in range(10):
        horizon_list += [i] * 10
    for i in range(10, 15):
        horizon_list += [i] * 20
    horizon_list += [15] * 5000  # the max steps it will reach
    if type(horizon) == torch.Tensor:
        return torch.Tensor(horizon_list, device=horizon.device)[horizon]
    elif type(horizon) == np.ndarray:
        return np.array(horizon_list)[horizon]
    elif type(horizon) == int:
        return horizon_list[horizon]
    else:
        assert False

class DatasetLoader(Dataset):
    
    def __init__(self, 
                 in_dir: Union[str, list], 
                 aug_ratio: float, 
                 embedding_dict: dict,  # goal embeddings
                 per_data_filters:list=None,
                 skip_frame: int=3,
                 window_len: int=20,
                 chunk_size: int=8,
                 padding_pos: str='left',
                 random_start: bool=True):
        
        super().__init__()
        if type(in_dir) == str:
            self.base_dirs = [in_dir]
        else:
            self.base_dirs = in_dir
        
        self.aug_ratio = aug_ratio
        self.embedding_dict = embedding_dict
        self.filters = list(embedding_dict.keys())  # goals
        self.skip_frame = skip_frame
        self.window_len = window_len
        self.chunk_size = chunk_size
        self.padding_pos = padding_pos  # left
        self.random_start = random_start  # True
        
        self.trajectories = {}  # {name: {'item': [val, ...], ...}, ...}
        self.goal_name, self.findcave_name = [], []
        for dir in self.base_dirs:
            i = 0
            for name in tqdm(os.listdir(dir)):
                # if i >= 100:
                #     break
                i += 1
                pickle_path = os.path.join(dir, name)
                with open(pickle_path, 'rb') as file:
                    traj_data = file.read()
                    traj = pickle.loads(traj_data)  # dict
                name = name.split('.')[0]
                if traj['goal'][0] in set(['log', 'sheep', 'cow', 'pig']):
                    self.goal_name.append(name)
                else:
                    self.findcave_name.append(name)
                self.trajectories[name] = traj
        print('---- Dateset loader initialized. ----')    

    def __len__(self):
        return self.aug_ratio  # 1e4 * 32

    def padding(self, goal, state, action, horizon, timestep):
        
        window_len = self.window_len
        traj_len = goal.shape[0]
        
        rgb_dim = state['rgb'].shape[1:]  # [traj_len, C, H, W] -> [C, H, W]
        voxels_dim = state['voxels'].shape[1:]
        compass_dim = state['compass'].shape[1:]
        gps_dim = state['gps'].shape[1:]
        biome_dim = state['biome'].shape[1:]
        
        action_dim = action.shape[1:]
        goal_dim = goal.shape[1:]
        
        if self.padding_pos == 'left':
            state['rgb'] = np.concatenate([np.zeros((window_len - traj_len, *rgb_dim)), state['rgb']], axis=0)
            state['voxels'] = np.concatenate([np.zeros((window_len - traj_len, *voxels_dim)), state['voxels']], axis=0)
            state['compass'] = np.concatenate([np.zeros((window_len - traj_len, *compass_dim)), state['compass']], axis=0)
            state['gps'] = np.concatenate([np.zeros((window_len - traj_len, *gps_dim)), state['gps']], axis=0)
            state['biome'] = np.concatenate([np.zeros((window_len - traj_len, *biome_dim)), state['biome']], axis=0)
            state['prev_action'] = np.concatenate([np.zeros((window_len - traj_len, *action_dim)), state['prev_action']], axis=0)
            goal = np.concatenate([np.zeros((window_len - traj_len, *goal_dim)), goal], axis=0)
            action = np.concatenate([np.zeros((window_len - traj_len, *action_dim)), action], axis=0)
            horizon = np.concatenate([np.zeros((window_len - traj_len)), horizon], axis=0)
            timestep = np.concatenate([np.zeros((window_len - traj_len)), timestep], axis=0)
            mask = np.concatenate([np.zeros((window_len - traj_len)), np.ones((traj_len))], axis=0)
            
        elif self.padding_pos == 'right':
            state['rgb'] = np.concatenate([state['rgb'], np.zeros((window_len - traj_len, *rgb_dim))], axis=0)
            state['voxels'] = np.concatenate([state['voxels'], np.zeros((window_len - traj_len, *voxels_dim))], axis=0)
            state['compass'] = np.concatenate([state['compass'], np.zeros((window_len - traj_len, *compass_dim))], axis=0)
            state['gps'] = np.concatenate([state['gps'], np.zeros((window_len - traj_len, *gps_dim))], axis=0)
            state['biome'] = np.concatenate([state['biome'], np.zeros((window_len - traj_len, *biome_dim))], axis=0)
            state['prev_action'] = np.concatenate([state['prev_action'], np.zeros((window_len - traj_len, *action_dim))], axis=0)
            goal = np.concatenate([goal, np.zeros((window_len - traj_len, *goal_dim))], axis=0)
            action = np.concatenate([action, np.zeros((window_len - traj_len, *action_dim))], axis=0)
            horizon = np.concatenate([horizon, np.zeros((window_len - traj_len))], axis=0)
            timestep = np.concatenate([timestep, np.zeros((window_len - traj_len))], axis=0)
            mask = np.concatenate([np.ones((traj_len)), np.zeros((window_len - traj_len))], axis=0)
        
        else:
            assert False
        
        state['rgb'] = torch.from_numpy(state['rgb']).float()
        state['voxels'] = torch.from_numpy(state['voxels']).long()
        state['compass'] = torch.from_numpy(state['compass']).float()
        state['gps'] = torch.from_numpy(state['gps']).float()
        state['biome'] = torch.from_numpy(state['biome']).long()
        state['prev_action'] = torch.from_numpy(state['prev_action']).float()
        action = torch.from_numpy(action).float()
        goal = torch.from_numpy(goal).float()
        horizon = torch.from_numpy(horizon).long()
        timestep = torch.from_numpy(timestep).long()
        mask = torch.from_numpy(mask).long()
        
        return goal, state, action, horizon, timestep, mask

    def __getitem__(self, idx):
        if random.choice([0, 1]) == 0:
            name = random.choice(self.goal_name)
        else:
            name = random.choice(self.findcave_name)
        traj_meta = self.trajectories[name]
        goal = traj_meta['goal'][0]
        n_frames = len(traj_meta['rgb'])

        assert n_frames > self.skip_frame
        '''
            sample video segment from action quanlity score, the higher scrore,
            more liky the segment will be chosen.'''
        aq = torch.from_numpy(traj_meta['action_quality'] + 0.01).float()
        cg = Categorical(aq)
        rand_start = cg.sample().item()
        ''' 
            start from at least 1 for sampling prev_action, end at most n_frame
            - skip_frame for at least one frame is chosen. '''
        rand_start = max(1, rand_start)
        rand_start = min(rand_start, n_frames - self.skip_frame)
        # the actual frames chosen for training
        snap_len = min((n_frames - rand_start) // self.skip_frame, self.window_len)
        frame_end = rand_start + snap_len * self.skip_frame

        state = {}
        state['rgb'] = traj_meta['rgb'][rand_start:frame_end:self.skip_frame]
        state['prev_action'] = traj_meta['action'][rand_start-1:frame_end-1:self.skip_frame]
        action = traj_meta['action'][rand_start:frame_end:self.skip_frame]
        # action out of range check
        action *= (action < np.array([[3, 3, 4, 11, 11, 8, 1, 1]]))
        state['prev_action'] *= (state['prev_action'] < np.array([[3, 3, 4, 11, 11, 8, 1, 1]]))


        if traj_meta.__contains__('voxels'):
            state['voxels'] = traj_meta['voxels'][rand_start:frame_end:self.skip_frame]
            state['compass'] = traj_meta['compass'][rand_start:frame_end:self.skip_frame]
            state['biome'] = traj_meta['biome'][rand_start:frame_end:self.skip_frame]
            state['gps'] = traj_meta['gps'][rand_start:frame_end:self.skip_frame] / np.array([[1000., 100., 1000.]])
        else:
            state['voxels'] = np.ones((snap_len, 3, 2, 2), dtype=np.int64)
            state['biome'] = np.ones((snap_len,), dtype=np.int64)
            state['compass'] = np.zeros((snap_len, 2), dtype=np.float32)
            state['gps'] = np.zeros((snap_len, 3), dtype=np.float32)
        
        goal = np.repeat(self.embedding_dict[goal], snap_len, 0)

        timestep = np.arange(0, snap_len)
        # the remaining steps
        horizon_list = np.arange(n_frames-rand_start-1, n_frames-frame_end-1, -self.skip_frame)
        horizon_list = discrete_horizon(horizon_list)
        
        return self.padding(goal, state, action, horizon_list, timestep)
        
