
"""
Taken from: https://github.com/LauraRuis/multimodal_seq2seq_gSCAN/blob/master/read_gscan/read_gscan.py
"""
import argparse
from enum import IntEnum
import json

import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Union

class Actions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2
    # Pick up & Drop an object
    pickup = 3
    drop = 4
    # Push & Pull
    push = 5
    pull = 6
    # Noop
    stay = 7
    # Done completing task
    done = 8

COMMANDS_TO_ACTIONS = {
    "turn left": Actions.left,
    "turn right": Actions.right,
    "walk": Actions.forward,
    "run": Actions.forward,
    "jump": Actions.forward,
    "push": Actions.push,
    "pull": Actions.pull,
    "stay": Actions.stay,
    "done": Actions.done
}

class gSCAN(Dataset):
    def __init__(self,  data_dir_path, split='test', use_images=False, use_actions=False, gen_action=True, combined_state_action_traj=False, max_traj_length=80):
        self.data_dir_path = data_dir_path
        self.data_file = None
        self.use_images = use_images
        self.max_traj_length = max_traj_length
        self.use_actions = use_actions
        self.combined_state_action_traj = combined_state_action_traj
        self.modalities = ['text']
        self.offsets = []
        if self.use_actions:
            if self.combined_state_action_traj:
                self.modalities.append('trajs')
            else:
                self.modalities.append('actions')
                self.modalities.append('states')
        elif gen_action:
            self.modalities.append('actions')
            self.modalities.append('init_state')
        else:
            self.modalities.append('states')
        self.set_split(split)

    def set_split(self, split):
        self.split = split
        with open(self.data_dir_path / f'{self.split}_offsets.txt', 'r') as offset_file:
            self.offsets = json.load(offset_file)

    def get_kwargs(self):
        with open(self.data_dir_path / f'{self.split}_data.txt', 'r') as data_file:
            data_file.seek(0)
            traj = data_file.read(self.offsets[1])
            _, _, states = json.loads(traj)
        return {'modalities': self.modalities, 'num_actions': len(Actions),
                'state_size': np.prod(np.shape(states[0])), 'max_seq_length': self.max_traj_length,
                'max_text_length': 64}

    def get_all_labels(self):
        return None, {}

    def collate_fn(self, batch):
        new_batch = {'text': {'input': [], 'labels': None}}
        if 'states' in self.modalities:
            new_batch['states'] = {'input': [], 'attention_mask': [], 'labels': None}
        if 'actions' in self.modalities:
            new_batch['actions'] = {'input': [], 'attention_mask': [], 'labels': None}
        if 'trajs' in self.modalities:
            new_batch['trajs'] = {'input': [], 'attention_mask': [], 'labels': None}
        if 'init_state' in self.modalities:
            new_batch['init_state'] = {'input': [], 'attention_mask': []}
        for sample in batch:
            for key in new_batch.keys():
                new_batch[key]['input'].append(sample[key]['input'])
                if key in ['actions', 'states', 'trajs']:
                    new_batch[key]['attention_mask'].append(sample[key]['attention_mask'])
        # Change necessary inputs from lists to numpy arrays
        for key in new_batch.keys():
            if key in ['actions', 'states', 'init_state']:
                new_batch[key]['input'] = np.array(new_batch[key]['input'])
                new_batch[key]['attention_mask'] = np.array(new_batch[key]['attention_mask'])
        if 'init_state' in self.modalities:
            new_batch['init_state'] = {'input': (new_batch['init_state']['input'], new_batch['text']['input']), 'labels': None}
        return new_batch

    def create_att_mask(self, size, mask_start):
        mask = np.zeros(size)
        mask[mask_start:] = 1
        return mask

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, item):
        with open(self.data_dir_path / f'{self.split}_data.txt', 'r') as data_file:
            data_file.seek(self.offsets[item])
            traj = data_file.read(self.offsets[item + 1] - self.offsets[item])
        mission, actions, states = json.loads(traj)
        mission = ' '.join(mission)
        # Pad actions
        np_actions = np.full(self.max_traj_length, Actions.stay, dtype=int)
        if len(actions) > self.max_traj_length:
            print(f'{len(actions)} > {self.max_traj_length}')
        index = min(len(actions), self.max_traj_length)
        np_actions[:index] = actions[:index]
        action_att_mask = self.create_att_mask(self.max_traj_length, index)
        # Pad states
        np_states = np.zeros((self.max_traj_length, *np.shape(states[0])), dtype=int)
        index = min(len(states), self.max_traj_length)
        np_states[:index] = states[:index]
        state_att_mask = self.create_att_mask(self.max_traj_length, index)

        sample = {'text': {'input': mission, 'labels': None}}
        if 'states' in self.modalities:
            sample['states'] = {'input': np_states, 'attention_mask': state_att_mask, 'labels': None}
        if 'actions' in self.modalities:
            sample['actions'] = {'input': np_actions, 'attention_mask': action_att_mask, 'labels': None}
        if 'trajs' in self.modalities:
            sample['trajs'] = {'input': (np_states, np_actions), 'attention_mask': (state_att_mask, action_att_mask), 'labels': None}
        if 'init_state' in self.modalities:
            sample['init_state'] = {'input': np.array(states[0])}
        return sample



if __name__ == "__main__":
    dataset = gSCAN(Path('test_datasets'), 'test')
    print(dataset[0])
