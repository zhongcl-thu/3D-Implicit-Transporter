import argparse
import json
import os
import pickle

import numpy as np
import torch
from tqdm import trange

from sim import PybulletSim
import sys 
sys.path.append('./')
from util import common, nets
from easydict import EasyDict as edict
from util.viz import *
from PIL import Image
from util.procrustes import rigid_transform_3d, pose_to_axis_angle
from pytorch3d.transforms import axis_angle_to_matrix


parser = argparse.ArgumentParser()

# global
parser.add_argument('--data_split', default='../mobility_dataset', type=str, help='path to split the dataset (the same path to training data)')
parser.add_argument('--test_data', default='./test_data', type=str, help='path to the testing data')
parser.add_argument('--model_path', default='./ckpts/model_best.pth', type=str, help='path to load the checkpoint')
parser.add_argument('--cfg_file', default='./config/condif.yaml', type=str, help='config file')
parser.add_argument('--save_path', default='./exp', type=str, help='path to save the results')
parser.add_argument('--mode', default='manipulation', type=str, choices=['exploration', 'manipulation'], help='type of test mode')
parser.add_argument('--seed', default=0, type=int, help='random seed of pytorch and numpy')
parser.add_argument('--strategy', default='largest', type=str)
parser.add_argument('--conditioned', default='goal', type=str) # goal init

# environment
parser.add_argument('--action_distance_ratio', default=10.0, type=float, help='dragging distance in each interaction')

# paper 
parser.add_argument('--paper_show', action='store_true', help='show')


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


def find_neareset_point(kp, pts):
    # kp = np.expand_dims(kp, 1).repeat(pts.shape[0], 1)
    dist_matrix = np.linalg.norm(pts[:, :3] - kp, axis=1)
    idx = dist_matrix.argmin(0)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    return [pts[idx, :3], np.asarray(pcd.normals)[idx], int(pts[idx, 3]*255.0 - 1)]


def normalize_pcd(pcd_data1, pcd_data2):
    bound_max = np.maximum(pcd_data1.max(0), pcd_data2.max(0)) #
    bound_min = np.minimum(pcd_data1.min(0), pcd_data2.min(0)) #

    center = (bound_min + bound_max) / 2
    scale = (bound_max - bound_min).max()

    pcd_data1 = (pcd_data1 - center) / scale
    pcd_data2 = (pcd_data2 - center) / scale

    return pcd_data1, pcd_data2, center, scale


def model_inference(pc_cur, pc_target, model, top_n=1, mode='largest', act_ratio=1.0, kp_final=None):
    pc_cur_xyz, pc_target_xyz, center, scale = normalize_pcd(pc_cur[:, :3], pc_target[:, :3])

    pc_cur_cuda = torch.unsqueeze(torch.from_numpy(pc_cur_xyz).float().cuda(), 0)
    pc_target_cuda = torch.unsqueeze(torch.from_numpy(pc_target_xyz).float().cuda(), 0)

    outputs = model(pc_cur_cuda, pc_target_cuda)
    
    center = torch.from_numpy(center).float().unsqueeze(0).unsqueeze(0).cuda()
    kps_cur = outputs['kps1'] / 2 * scale + center
    kps_target = outputs['kps2'] / 2  * scale + center

    if kp_final is not None:
        kps_cur = kps_target
        kps_target = kp_final
        pc_cur = pc_target

    T_cur2target, _ = rigid_transform_3d(kps_cur, kps_target)
    angles_cur2target, axis_cur2target_norm = pose_to_axis_angle(T_cur2target)

    if angles_cur2target > 0.1: # rad
        rot_cur2next = axis_angle_to_matrix(axis_cur2target_norm * 0.1) # B, 3, 3
        kps_cur_mean = kps_cur - kps_cur.mean(1)
        kps_next_pred = (rot_cur2next[:, :3, :3] @ kps_cur_mean.transpose(2, 1)).transpose(2, 1)

        dir_next2cur = (kps_next_pred - kps_cur_mean)[0].cpu().numpy()
    else:
        pass
    
    kps_cur = kps_cur[0].cpu().numpy()
    kps_target = kps_target[0].cpu().numpy()
    
    # custom_draw_geometry_with_key_callback([downpcd], save_path='./test.png')
    dir_tar2cur = kps_target - kps_cur # 6,3
    dir_tar2cur_dis = np.linalg.norm(dir_tar2cur, 2, 1, keepdims=True)
    dir_tar2cur_norm = dir_tar2cur / dir_tar2cur_dis #6, 3
    
    if mode == 'largest':
        idx = np.argsort(dir_tar2cur_dis[:, 0])[-top_n]
    elif mode == 'voting':
        dir_tar2cur_norm_sum = (dir_tar2cur_norm @ dir_tar2cur_norm.T).sum(1)
        idx = np.argsort(dir_tar2cur_norm_sum)[-top_n]
    
    if angles_cur2target > 0.1: 
        dir_next2cur_idx = dir_next2cur[idx]
        direction_next_idx_norm = dir_next2cur_idx / np.linalg.norm(dir_next2cur_idx)

        if dir_tar2cur_norm[idx].T @ direction_next_idx_norm < 0:
            direction_next_idx_norm = -direction_next_idx_norm
        manipu_dir = direction_next_idx_norm * dir_tar2cur_dis[idx, 0] * act_ratio
        
    else:
        manipu_dir = dir_tar2cur[idx] * act_ratio

    anchor = kps_cur[idx].copy()
    anchor += manipu_dir
    anchor = np.expand_dims(anchor, 0)
    
    pos_normal_link_id = find_neareset_point(kps_cur[idx], pc_cur)

    return pos_normal_link_id, manipu_dir, kps_cur, kps_target


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
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    
    split_meta = json.load(open(os.path.join(args.data_split, 'split-full.json'), 'r'))

    # Load model
    device = torch.device(f'cuda:0')
    C = common.Config(args.cfg_file)
    config_common = edict(C.config["common"])
    config_public = config_common.public_params

    model = nets.model_entry(config_common.net, config_public)
    try:
        model.load_state_dict(torch.load(args.model_path))
    except:
        checkpoint = torch.load(args.model_path, lambda a,b:a)
        model.load_state_dict({k[7:]:v for k,v in checkpoint.items()})
    
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    pool_list = list()
    for category_type in ['train', 'test']: #
        for category_name in split_meta[category_type].keys():
            instance_type = 'test'
            pool_list.append((category_type, category_name, instance_type))
    
    dists = {'train': 0, 'train_len': len(split_meta['train']), 
            'test': 0, 'test_len': len(split_meta['test'])}
    success_rates = {'train': 0, 'test': 0}

    for category_type, category_name, instance_type in pool_list:
        if category_name not in ['Toy', 'Switch', 'Bucket']:
            dist, succ_rate = run_test(args, model, category_name, instance_type)
            dists[category_type] += dist
            success_rates[category_type] += succ_rate
    
    with open(os.path.join(args.save_path, 'manipulation_result.txt'), 'a', encoding='utf-8') as f:
        f.write('Mean train----{:.3f}-----{:.3f} \n'.format(dists['train']/dists['train_len'], 
                                                    success_rates['train']/dists['train_len']))
        f.write('Mean test----{:.3f}-----{:.3f} \n'.format(dists['test']/dists['test_len'], 
                                                    success_rates['test']/dists['test_len']))


def run_test(args, model, category_name, instance_type):
    print(f'==> run test: {args.mode} - {category_name} - {instance_type}')

    max_step_num = step_num_dict[category_name]

    # test data info
    test_data_path = os.path.join(args.test_data, args.mode, category_name, instance_type)
    test_num = 100

    results = dict()
    sim = PybulletSim(True, paper_show=args.paper_show) #args.action_distance, 
    seq_with_correct_position = list()
    joint_type_list = list()

    cnt = 0

    for id in trange(0, test_num): # 44
        # Reset random seeds
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        scene_state = pickle.load(open(os.path.join(test_data_path, f'{id}.pkl'), 'rb'))
        observation = sim.reset(scene_state=scene_state, save_data=False, mobility_path=args.data_split) # N, 4
        # position/direction inference
        pos_normal_link_id, direction, kp_cur, kp_final = model_inference(observation['pc_cur_3cam'], 
                                                        observation['pc_init_3cam'], 
                                                        model, mode=args.strategy, 
                                                        act_ratio=args.action_distance_ratio)
        obs_cur_root = observation['pc_cur_3cam'].copy()
        kp_final = torch.from_numpy(kp_final).unsqueeze(0).cuda()
        if args.paper_show:
            obs_pcd = sim.fusion_multi_cam_pcd()
            viz_pc_keypoint(obs_pcd[:, :3], kp_cur, 0.03, 
                            save_path='ours/results/paper_show/{}_{}_{}.png'.format(category_name, id, 0))

        observation, done, info = sim.step([0, pos_normal_link_id, direction, np.zeros(3), np.zeros(3)])
        # pos_normal_link_id[-1] == 3
        # observation, done, info = sim.step([0, pos_normal_link_id, direction, np.zeros(3), np.zeros(3)])
        # terminate immediately if the position is wrong
        if done:
            results[f'sequence-{id}'] = -1234 # specific constant for wrong position
            joint_type_list.append(None)
            continue
        else:
            seq_with_correct_position.append(id)
            joint_type_list.append(sim.get_joint_type())

        # pre-preparation
        reach_init = False
        # bad_actions = list()
        dist2target = info['dist2init']
        results[f'dist2target-{id}-{0}'] = dist2target

        top_n = 1

        direction_sum = np.zeros(3)
        for step in range(1, max_step_num + 1):
            if reach_init:
                results[f'dist2target-{id}-{step}'] = 0
                continue

            # find whether state change
            if np.sum(np.abs(observation['last_image'] - observation['cur_image']) > 1e-4) < 2500:
                top_n += 1
                if top_n > 6:
                    results[f'sequence-{id}'] = step
                    results[f'dist2target-{id}-{max_step_num}'] = info['dist2init']
                    break
                
                direction_sum = np.zeros(3)
            else:
                top_n = 1
                direction_sum += direction
            
            direction_last = direction

            # position/direction inference
            if args.conditioned == 'goal':
                pos_normal_link_id, direction, kp_cur, _ = model_inference(observation['pc_cur_3cam'],
                                                        observation['pc_init_3cam'], 
                                                        model, top_n, mode=args.strategy,
                                                        act_ratio=args.action_distance_ratio)
            else:
                pos_normal_link_id, direction, kp_cur, _ = model_inference(obs_cur_root, 
                                                        observation['pc_cur_3cam'],
                                                        model, top_n, mode=args.strategy,
                                                        act_ratio=args.action_distance_ratio,
                                                        kp_final=kp_final)
            
            
            if args.paper_show:
                obs_pcd = sim.fusion_multi_cam_pcd()
                viz_pc_keypoint(obs_pcd[:, :3], kp_cur, 0.03, 
                                save_path='ours/results/paper_show/{}_{}_{}.png'.format(category_name, id, step))

            #! direction_dif?
            direction_dif = direction - direction_last
            observation, reach_init, info = sim.step([1, pos_normal_link_id, direction, direction_dif, direction_sum])

            dist2target = info['dist2init']
            results[f'dist2target-{id}-{step}'] = info['dist2init']
            if reach_init:
                results[f'sequence-{id}'] = step
        
               #! add maximum=1
        if results[f'dist2target-{id}-{max_step_num}'] > results[f'dist2target-{id}-{0}']:
            results[f'dist2target-{id}-{max_step_num}'] = results[f'dist2target-{id}-{0}']
        if results[f'dist2target-{id}-{0}'] == 0:
            print('inf')
        else:
            print(results[f'dist2target-{id}-{max_step_num}']/results[f'dist2target-{id}-{0}'])

    # result analysis
    final_result = 0
    success = 0
    # manipulationfor id in range(test_num):
    for id in range(test_num):
        if id in seq_with_correct_position:
            if results[f'dist2target-{id}-{0}'] != 0:
                final_result += results[f'dist2target-{id}-{max_step_num}'] / results[f'dist2target-{id}-{0}'] / test_num
                if results[f'dist2target-{id}-{max_step_num}'] / results[f'dist2target-{id}-{0}'] < 0.1:
                    success += 1
            else:
                final_result += 1.0/test_num

        else:
            final_result += 1.0 / test_num

    success /= test_num

    print(f'{args.mode} results (distance) - {category_name}-{instance_type}: {final_result}')
    print(f'{args.mode} results (success) - {category_name}-{instance_type}: {success}')

    with open(os.path.join(args.save_path, 'manipulation_result.txt'), 'a', encoding='utf-8') as f:
        f.write('{}----{:.3f}-----{:.3f} \n'.format(category_name, final_result, success))
    
    return final_result, success


if __name__=='__main__':
    main()
