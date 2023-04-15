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
        for dir in self.base_dirs:
            img_dir = os.path.join(dir, 'video-sample')
            lmdb_dir = os.path.join(dir, 'lmdb-test')
            env = lmdb.open(lmdb_dir)
            txn = env.begin()
            for key, value in tqdm(txn.cursor()):
                name = key.decode()
                traj = pickle.loads(value)  # dict
                traj['rgb'] = []
                imholder = ImageHolder()
                videoholder = VideoHolder(os.path.join(img_dir, name + '.mp4'))
                for frame in videoholder.read_frame():
                    frame = imholder.hwc2chw(frame[..., ::-1])  # [H, W, BGR] -> [RGB, H, W]
                    traj['rgb'].append(frame)
                if len(traj['rgb']) == 0:  # empty list is not allowed for np.stack
                    continue
                traj['rgb'] = np.stack(traj['rgb'])
                self.trajectories[name] = traj
                

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
        name = random.choice(list(self.trajectories.keys()))
        traj_meta = self.trajectories[name]
        goal = traj_meta['goal'][0]
        horizon = traj_meta['horizon'][0]  # total length
        if horizon != len(traj_meta['rgb']):
            print('Horizon not correct in id: {}'.format(name))
            horizon = len(traj_meta['rgb'])

        assert horizon > self.skip_frame
        # always rand_start:
        traj_meta['action_quality']
        aq = torch.from_numpy(traj_meta['action_quality'])
        cg = Categorical(aq)
        rand_start = cg.sample().item()
        rand_start = max(1, rand_start)
        rand_start = min(rand_start, horizon - self.skip_frame - 1)
        # rand_start = random.randint(1, horizon - self.skip_frame)
        # snap_len >= 1
        snap_len = min((horizon - rand_start) // self.skip_frame, self.window_len)
        frame_end = rand_start + snap_len * self.skip_frame


        state = {}
        state['rgb'] = traj_meta['rgb'][rand_start:frame_end:self.skip_frame]
        state['voxels'] = traj_meta['voxels'][rand_start:frame_end:self.skip_frame]
        state['compass'] = traj_meta['compass'][rand_start:frame_end:self.skip_frame]
        state['gps'] = traj_meta['gps'][rand_start:frame_end:self.skip_frame] / np.array([[1000., 100., 1000.]])
        
        state['biome'] = traj_meta['biome'][rand_start:frame_end:self.skip_frame]
        state['prev_action'] = traj_meta['action'][rand_start-1:frame_end-1:self.skip_frame]
        

        action = traj_meta['action'][rand_start:frame_end:self.skip_frame]


        goal = np.repeat(self.embedding_dict[goal], snap_len, 0)

        timestep = np.arange(0, snap_len)
        # the remaining steps
        horizon_list = np.arange(horizon-rand_start-1, horizon-frame_end-1, -self.skip_frame)
        horizon_list = discrete_horizon(horizon_list)
        
        return self.padding(goal, state, action, horizon_list, timestep)
        
    