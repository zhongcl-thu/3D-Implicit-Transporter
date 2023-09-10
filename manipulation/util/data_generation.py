import argparse
import json
import os
import pickle

import numpy as np
import torch

import spherical_sampling
import train
from model import Model
from sim import PybulletSim
import ipdb
import tqdm
from tqdm import trange


parser = argparse.ArgumentParser()

# global
parser.add_argument('--checkpoint', default='pretrained/umpnet.pth', type=str, help='path to the checkpoint')
parser.add_argument('--mode', default='manipulation', type=str, choices=['exploration', 'manipulation'], help='type of test mode')
parser.add_argument('--seed', default=0, type=int, help='random seed of pytorch and numpy')

# model
parser.add_argument('--model_type', default='sgn_mag', type=str, choices=['sgn', 'mag', 'sgn_mag'], help='model_type')

# environment
parser.add_argument('--num_direction', default=64, type=int, help='number of directions')
parser.add_argument('--no_cem', action='store_true', help='without cem')
parser.add_argument('--action_distance', default=0.18, type=float, help='dragging distance in each interaction')


step_num_dict = {
    'Refrigerator': 12,
    'FoldingChair': 8,
    'Laptop': 12,
    'Stapler': 15,
    'TrashCan': 9,
    'Microwave': 8,
    'Toilet': 7,
    'Window': 6,
    'StorageFurniture': 9,
    'Switch': 7,
    'Kettle': 3,
    'Toy': 10,
    'Box': 10,
    'Phone': 12,
    'Dishwasher': 10,
    'Safe': 10,
    'Oven': 9,
    'WashingMachine': 9,
    'Table': 7,
    'KitchenPot': 3,
    'Bucket': 13,
    'Door': 10
}


def calc_novel_state_ratio(visited):
    visited = np.sort(visited)
    last_val = -1000000 # -inf
    cnt = 0
    for val in visited:
        if val - last_val > 0.15:
            cnt += 1
            last_val = val
    ratio = cnt / len(visited)
    
    return ratio


def main():
    args = parser.parse_args()

    mobility_path = 'mobility_dataset'
    split_file = 'split-full.json' #
    split_meta = json.load(open(os.path.join(mobility_path, split_file), 'r'))

    pool_list = list()
    for category_type in ['train', 'test']: #, 'test'
        for category_name in split_meta[category_type].keys():
            if category_name == 'Kettle':
                instance_type = 'train' # 'train'
                pool_list.append((category_type, category_name, instance_type))
    
    for category_type, category_name, instance_type in pool_list:
        data_gen(args, category_type, category_name, instance_type, split_meta)

def data_gen(args, category_type, category_name, instance_type, split_meta):
    print(f'==> run test: {args.mode} - {category_name} - {instance_type}')

    sim = PybulletSim(True, args.action_distance)
    max_step_num = step_num_dict[category_name]

    # test data info
    test_data_path = os.path.join('test_data', args.mode, category_name, instance_type)
    test_num = 100
    
    print('***********start cat:{}'.format(category_name))
    
    if instance_type == 'test':
        for id in trange(test_num):
            # Reset random seeds
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            
            scene_state = pickle.load(open(os.path.join(test_data_path, f'{id}.pkl'), 'rb'))
            
            out_dir = 'bullet_mobility_data_test/{}/{}'.format(category_name, id)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            observation = sim.reset(scene_state=scene_state, save_data=True, out_dir=out_dir)
    else:
        for instance_id in tqdm.tqdm(split_meta[category_type][category_name][instance_type]):
            # Reset random seeds
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
                
            sim.reset(scene_state=None, 
                        category_type=category_type, 
                        instance_type=instance_type,
                        category_name=category_name,
                        instance_id=instance_id)
            
            out_dir = 'bullet_mobility_data/{}/{}'.format(category_name, instance_id)
            if  not os.path.exists(out_dir):
                os.makedirs(out_dir)

            sim.change_state(out_dir)


if __name__=='__main__':
    main()