import numpy as np
import os
import ipdb
from tqdm import trange
import pickle
import json

data_path = 'data/bullet_mobility_data_test'

mobility_path = 'mobility_dataset'
split_file = 'split-full.json' #
split_meta = json.load(open(os.path.join(mobility_path, split_file), 'r'))

pool_list = list()
for category_type in ['train', 'test']: #, 'test'
    for category_name in split_meta[category_type].keys():
        instance_type = 'test' # 'train'
        
        test_data_path = os.path.join('test_data', 'manipulation', category_name, instance_type)
        for id in trange(100):
            scene_state = pickle.load(open(os.path.join(test_data_path, f'{id}.pkl'), 'rb'))
            
            out_dir = 'bullet_mobility_data_test/{}/{}'.format(category_name, id)
            np.savetxt(os.path.join(out_dir, 'select_joint_id.txt'), np.array([scene_state['selected_joint']]), fmt='%d')
