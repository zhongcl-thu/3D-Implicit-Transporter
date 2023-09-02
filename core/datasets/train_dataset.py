import numpy as np
import os
import h5py
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

from core.datasets.augmentation import atomic_rotate
from core.utils.viz import *


def make_3d_grid(x_res, y_res, z_res, type='DHW'):
    cor_x = np.linspace(x_res[0], x_res[1], x_res[2])
    cor_y = np.linspace(y_res[0], y_res[1], y_res[2])
    cor_z = np.linspace(z_res[0], z_res[1], z_res[2])

    
    if type == 'DHW':
        Z, Y, X = np.meshgrid(cor_y, cor_z, cor_x)
        grid = np.stack((Y, Z, X)).reshape(3, -1).T
        
        tmp = np.copy(grid[:, 0])
        grid[:, 0] = grid[:, 2]
        grid[:, 2] = tmp
    
    elif type == 'HWD':
        X, Y, Z = np.meshgrid(cor_y, cor_x, cor_z)
        grid = np.stack((Y, X, Z)).reshape(3, -1).T
    
    return grid


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


class BaseDataset(Dataset):
    def __init__(self, config, mode):
        Dataset.__init__(self)
        
        self.mode = mode

        config_data = config["data_info"]
        config_public_params = config["common"]["public_params"]
        self.config_public_params = config_public_params
        self.config_aug = config["augmentation"]

        if mode == 'train':
            file_name = config_data["train_file"]
        elif mode == 'val':
            file_name = config_data["val_file"]
        elif mode == 'test':
            file_name = config_data["test_file"]

        with open(os.path.join(os.getcwd(), file_name), 'r') as f:
            self.filenames = f.read().splitlines()

        self.pcd_root = config_data["data_path"]
        self.input_num = config_public_params["input_pcd_num"]
        self.on_occupancy_num = config_public_params["on_occupancy_num"]
        self.off_occupancy_num = config_public_params["off_occupancy_num"]

    def __len__(self):
        return len(self.filenames)

    def get_key(self, idx, suffix):
        return self.filenames[idx] + suffix

    def __getitem__(self, idx):
        raise NotImplementedError()


class Articulated_Obj_Syn(BaseDataset):
    def __init__(self, config, mode):
        super().__init__(config, mode)

        self.config = config
        self.padding = self.config_public_params['padding']
        self.bound = 0.5 * (self.padding + 1)
        self.max_down_sample = self.config_aug.get('max_down_sample', 1)
        self.pc_length = 30
        # self.dif_length = self.config_aug['dif_length']
        self.get_gt_pose = self.config['data_info'].get('gt_pose', False)

        self.sampling_mode = self.config_public_params.get('sampling_mode', 'random')
    

    def get_input(self, index):
        # get pcd
        obj_name, articulate_type, instance_name, name_index = self.filenames[index].split(' ')
        name_index = int(name_index)

        # temporal difference
        low_dif, high_dif = np.random.randint(5, 15, size=2)
        name_index_1 = name_index - low_dif
        name_index_2 = name_index + high_dif
        if name_index_2 > self.pc_length:
            name_index_2 -= self.pc_length

        data = np.load(os.path.join(self.pcd_root, obj_name, instance_name, 'pc.npy'), 
                        allow_pickle=True)
        
        if self.get_gt_pose:
            pose = np.load(os.path.join(self.pcd_root, obj_name, instance_name, 'part_pose.npy'), 
                            allow_pickle=True)

            pose21 = pose[name_index] @ np.linalg.inv(pose[name_index_1]) 
            pose32 = pose[name_index_2] @ np.linalg.inv(pose[name_index])
        
            return  data[name_index_1][:, :3].astype(np.float), \
                    data[name_index][:, :3].astype(np.float), \
                    data[name_index_2][:, :3].astype(np.float), \
                    pose21.astype(np.float), pose32.astype(np.float), \
                    obj_name, articulate_type
        else:
            return data[name_index_1][:, :3].astype(np.float), \
                    data[name_index][:, :3].astype(np.float), \
                    data[name_index_2][:, :3].astype(np.float), \
                    None, None, \
                    obj_name, instance_name, articulate_type


    def normalize_pcd(self, pcd_data):
        pcd_data -= pcd_data.mean(0)

        dis = np.linalg.norm(pcd_data, axis=1)
        scale = dis.max()
        
        pcd_data /= (scale + 1e-5) # -1 ~ 1
        pcd_data /= 2 # -0.5 ~ 0.5

        return pcd_data


    def prepare_input_data(self, pcd_data, max_down_sample=1):
        if max_down_sample > 1:
            down_sample = np.random.uniform(1, max_down_sample, size=1)
        else:
            down_sample = 1
        
        rand_idcs = np.random.choice(pcd_data.shape[0], 
                                    size=int(self.input_num / down_sample), replace=True)
        pcd_data_s = pcd_data[rand_idcs]

        if rand_idcs.shape[0] < self.input_num:
            fix_idx = np.asarray(range(pcd_data_s.shape[0]))
            while pcd_data_s.shape[0] + fix_idx.shape[0] < self.input_num:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(pcd_data_s.shape[0]))), 
                                            axis=0)
            random_idx = np.random.choice(pcd_data_s.shape[0], 
                                            self.input_num - fix_idx.shape[0], 
                                            replace=False)
            rand_idcs = np.concatenate((fix_idx, random_idx), axis=0)
        
            pcd_data_s = pcd_data_s[rand_idcs]
        
        return pcd_data_s


    def check_input(self, pcd, modify=True):
        valid_mask_xy = np.logical_and(np.abs(pcd)[:, 0] < self.bound_x,
                                        np.abs(pcd)[:, 1] < self.bound_x)
        valid_mask_z = np.logical_and(pcd[:, 2] < self.bound_z_max, 
                                        pcd[:, 2] > self.bound_z_min)
        valid_mask = valid_mask_xy * valid_mask_z

        if modify:
            return pcd[valid_mask]
        else:
            return pcd, valid_mask


    def prepare_occupancy_data(self, pcd_data, sampling_mode='random'):
        can_repeat = True if self.on_occupancy_num > pcd_data.shape[0] else False
        
        rand_idcs_on = np.random.choice(pcd_data.shape[0], 
                                        size=self.on_occupancy_num, 
                                        replace=can_repeat)
        on_surface_coords = pcd_data[rand_idcs_on]
        on_surface_labels = np.ones(self.on_occupancy_num)

        if sampling_mode == 'random':
            off_surface_x = np.random.uniform(-self.bound, self.bound, 
                                                size=(self.off_occupancy_num, 1))
            off_surface_y = np.random.uniform(-self.bound, self.bound, 
                                                size=(self.off_occupancy_num, 1))
            off_surface_z  = np.random.uniform(-self.bound, self.bound, 
                                                size=(self.off_occupancy_num, 1))
            off_surface_coords = np.concatenate((off_surface_x, off_surface_y, off_surface_z), 
                                                axis=1)
            off_surface_labels = np.zeros(self.off_occupancy_num)
        else:
            grid = make_3d_grid([-0.5, 0.5, 15], [-0.5, 0.5, 15], [-0.5, 0.5, 15])
            rand_idcs_on = np.random.choice(grid.shape[0], 
                                        size=self.off_occupancy_num)
            off_surface_coords = grid[rand_idcs_on]
            off_surface_labels = np.zeros(self.off_occupancy_num)
        
        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        labels = np.concatenate((on_surface_labels, off_surface_labels), axis=0)

        rix = np.random.permutation(coords.shape[0])
        coords = coords[rix]
        labels = labels[rix]

        return coords, labels
    

    def prepare_test_data(self,):
        z_max = self.config_public_params.get('z_max', 0.5)
        z_min = self.config_public_params.get('z_min', -0.5)
        
        coords = make_3d_grid([-0.5, 0.5, self.config['test']['grid_kwargs']['x_res']],
                            [-0.5, 0.5, self.config['test']['grid_kwargs']['y_res']],
                            [z_min, z_max, self.config['test']['grid_kwargs']['z_res']],
                            type='HWD')
        labels = coords

        return coords, labels


    def __getitem__(self, index):
        pc_start_ori, pc_middle_ori, pc_end_ori, pose21, pose32, \
            obj_name, instance_name, articulate_type = self.get_input(index)

        # sample
        pc_start = self.prepare_input_data(pc_start_ori, self.max_down_sample)
        pc_middle = self.prepare_input_data(pc_middle_ori, 1)
        pc_end = self.prepare_input_data(pc_end_ori, self.max_down_sample)
        
        if self.config_aug["do_aug"]:
            # rotate
            z_angle = np.random.uniform() * self.config_aug["rotate_angle"] / 180.0 * (np.pi)
            angles_2d = [0, 0, z_angle]

            pc_start = atomic_rotate(pc_start, angles_2d)
            pc_middle = atomic_rotate(pc_middle, angles_2d)
            pc_middle_ori = atomic_rotate(pc_middle_ori, angles_2d)
            pc_end = atomic_rotate(pc_end, angles_2d)

            # jitter (Gaussian noise)
            sigma, clip = self.config_aug["sigma"], self.config_aug["clip"]
            jitter1 = np.clip(sigma * np.random.uniform(pc_start.shape[0], 3), -1 * clip, clip)
            jitter2 = np.clip(sigma * np.random.uniform(pc_middle.shape[0], 3), -1 * clip, clip)
            jitter3 = np.clip(sigma * np.random.uniform(pc_end.shape[0], 3), -1 * clip, clip)
            pc_start += jitter1
            pc_middle += jitter2
            pc_end += jitter3

        # normalize
        bound_max = np.maximum(pc_start.max(0), pc_middle.max(0), pc_end.max(0)) #
        bound_min = np.minimum(pc_start.min(0), pc_middle.min(0), pc_end.min(0)) #
        center = (bound_min + bound_max) / 2
        scale = (bound_max - bound_min).max() # / (1 + self.padding)

        pc_start = (pc_start - center) / scale
        pc_middle = (pc_middle - center) / scale
        pc_middle_ori = (pc_middle_ori - center) / scale
        pc_end = (pc_end - center) / scale

        # get occp coords and labels
        if self.mode != 'test':
            occup_coords, occup_labels = self.prepare_occupancy_data(pc_middle_ori, self.sampling_mode)
        else:
            occup_coords, occup_labels = self.prepare_test_data()
        
        if articulate_type == 'R':
            joint_label = np.ones(1)
        else:
            joint_label = np.zeros(1)
        
        pc_start = torch.from_numpy(pc_start).float()
        pc_middle = torch.from_numpy(pc_middle).float()
        pc_end = torch.from_numpy(pc_end).float()
        occup_coords = torch.from_numpy(occup_coords).float()
        occup_labels = torch.from_numpy(occup_labels).float()
        joint_type_label = torch.from_numpy(joint_label).float()

        if self.get_gt_pose:
            pose21 = torch.from_numpy(pose21).float()
            pose32 = torch.from_numpy(pose32).float()
            
            center = torch.from_numpy(center).float()
            scale = torch.from_numpy(np.array([scale])).float()
        
            inputs = {'point_cloud_1': pc_start, 'point_cloud_2': pc_middle, 'point_cloud_3': pc_end, 
                    'coords_2': occup_coords, 'occup_labels_2': occup_labels,
                    'obj_name': obj_name, "gt_joint": joint_type_label,
                    'pose21': pose21, "pose32": pose32, "center": center, "scale": scale} 
        
        else:
            inputs = {'point_cloud_1': pc_start, 'point_cloud_2': pc_middle, 'point_cloud_3': pc_end, 
                    'coords_2': occup_coords, 'occup_labels_2': occup_labels,
                    'obj_name': obj_name, "instance_name": instance_name,
                    "gt_joint": joint_type_label, "center": center, "scale": scale
            }

        return inputs


class Articulated_Real_Obj(Articulated_Obj_Syn):
    def __init__(self, config, mode):
        super().__init__(config, mode)
        self.pc_length = 281


    def filter_points(self, pts):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(pts[:, 3:]/255.0)
        # o3d.visualization.draw_geometries([pcd])

        down_pcd = pcd.voxel_down_sample(voxel_size=self.config_aug['voxel_size'])
        cl, ind = down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # o3d.visualization.draw_geometries([cl])

        return np.concatenate((np.array(cl.points), np.array(cl.colors)), 1)


    # def get_input(self, index):
    #     # get pcd
    #     name_index, _ = self.filenames[index].split('.')
    #     name_index = int(name_index)

    #     # temporal difference
    #     low_dif, high_dif = np.random.randint(10, 30, size=2)
    #     name_index_1 = name_index - low_dif
    #     name_index_2 = name_index + high_dif
        
    #     name_index_1 = name_index_1 if name_index_1 > 0 else 0
    #     name_index_2 = name_index_1 if name_index_1 > 0 else 0

    #     data = np.load(os.path.join(self.pcd_root, '{:0>4d}.npy'.format(name_index)), 
    #                     allow_pickle=True).astype(np.float)
    #     data = self.filter_points(data)
        
    #     if name_index_1 < 0:
    #         name_index_1 = 0
        
    #     if name_index_2 > self.pc_length - 1:
    #         name_index_2 = self.pc_length - 1

    #     data1 = np.load(os.path.join(self.pcd_root, '{:0>4d}.npy'.format(name_index_1)), 
    #                     allow_pickle=True).astype(np.float)
    #     data1 = self.filter_points(data1)

    #     data2 = np.load(os.path.join(self.pcd_root, '{:0>4d}.npy'.format(name_index_2)), 
    #                     allow_pickle=True).astype(np.float)
    #     data2 = self.filter_points(data2)
        
    #     return data1, data, data2

    def get_input(self, index):
        # get pcd
        cat_name, instance_name = self.filenames[index].split(' ')

        # temporal difference
        low_dif, high_dif = np.random.randint(8, 25, size=2)
        index_1 = index - low_dif
        index_2 = index + high_dif

        if index_1 < 0:
            index_1 = 0
        if index_2 > len(self.filenames) - 1:
            index_2 = len(self.filenames) - 1
        
        cat_name_1, instance_name_1 = self.filenames[index_1].split(' ')
        if cat_name_1 != cat_name:
            cat_name_1, instance_name_1 = self.filenames[index + low_dif].split(' ')

        cat_name_2, instance_name_2 = self.filenames[index_2].split(' ')
        if cat_name_2 != cat_name:
            cat_name_2, instance_name_2 = self.filenames[index - high_dif].split(' ')

        data = np.loadtxt(os.path.join(self.pcd_root, '{}/pc/txt/{}'.format(cat_name, instance_name)))
        data = self.filter_points(data)

        data1 = np.loadtxt(os.path.join(self.pcd_root, '{}/pc/txt/{}'.format(cat_name_1, instance_name_1)))
        data1 = self.filter_points(data1)

        data2 = np.loadtxt(os.path.join(self.pcd_root, '{}/pc/txt/{}'.format(cat_name_2, instance_name_2)))
        data2 = self.filter_points(data2)
        
        return data1, data, data2


    def __getitem__(self, index):
        pc_middle_ori, pc_start_ori, pc_end_ori = self.get_input(index)

        # sample
        pc_start = self.prepare_input_data(pc_start_ori, self.max_down_sample)
        pc_middle = self.prepare_input_data(pc_middle_ori, 1)
        pc_end = self.prepare_input_data(pc_end_ori, self.max_down_sample)
        
        # ipdb.set_trace()
        # pc_start_show = make_o3d_pcd(pc_start[:, :3], pc_start[:, 3:])
        # pc_middle_show = make_o3d_pcd(pc_end[:, :3], pc_end[:, 3:])
        # o3d.visualization.draw_geometries([pc_start_show, pc_middle_show])

        if self.config_aug["do_aug"]:
            # rotate
            z_angle = np.random.uniform() * self.config_aug["rotate_angle"] / 180.0 * (np.pi)
            angles_2d = [0, z_angle, 0]

            pc_start_pts = atomic_rotate(pc_start[:, :3], angles_2d)
            pc_middle_pts = atomic_rotate(pc_middle[:, :3], angles_2d)
            pc_middle_ori_pts = atomic_rotate(pc_middle_ori[:, :3], angles_2d)
            pc_end_pts = atomic_rotate(pc_end[:, :3], angles_2d)

            # pc_start_show = make_o3d_pcd(pc_start_pts, pc_start[:, 3:])
            # pc_middle_show = make_o3d_pcd(pc_end_pts[:, :3], pc_end[:, 3:])
            # o3d.visualization.draw_geometries([pc_start_show, pc_middle_show])

            # jitter (Gaussian noise)
            sigma, clip = self.config_aug["sigma"], self.config_aug["clip"]
            jitter1 = np.clip(sigma * np.random.uniform(pc_start.shape[0], 3), -1 * clip, clip)
            jitter2 = np.clip(sigma * np.random.uniform(pc_middle.shape[0], 3), -1 * clip, clip)
            jitter3 = np.clip(sigma * np.random.uniform(pc_end.shape[0], 3), -1 * clip, clip)
            pc_start_pts += jitter1
            pc_middle_pts += jitter2
            pc_end_pts += jitter3
        else:
            pc_start_pts = pc_start[:, :3]
            pc_middle_pts = pc_middle[:, :3]
            pc_end_pts = pc_end[:, :3]
            pc_middle_ori_pts = pc_middle_ori[:, :3]


        # normalize
        bound_max = np.maximum(pc_start_pts.max(0), pc_middle_pts.max(0), pc_end_pts.max(0)) #
        bound_min = np.minimum(pc_start_pts.min(0), pc_middle_pts.min(0), pc_end_pts.min(0)) #
        center = (bound_min + bound_max) / 2
        scale = (bound_max - bound_min).max() # / (1 + self.padding)

        pc_start_pts = (pc_start_pts - center) / scale
        pc_middle_pts = (pc_middle_pts - center) / scale
        pc_middle_ori_pts = (pc_middle_ori_pts - center) / scale
        pc_end_pts = (pc_end_pts - center) / scale

        # get occp coords and labels
        if self.mode != 'test':
            occup_coords, occup_labels = self.prepare_occupancy_data(pc_middle_ori_pts)
        else:
            occup_coords, occup_labels = self.prepare_test_data()
        
        # ipdb.set_trasce()
        # pc_start = np.concatenate((pc_start_pts, pc_start[:, 3:]), 1)
        # pc_middle = np.concatenate((pc_middle_pts, pc_middle[:, 3:]), 1)
        # pc_end = np.concatenate((pc_end_pts, pc_end[:, 3:]), 1)

        pc_start = pc_start_pts
        pc_middle = pc_middle_pts
        pc_end = pc_end_pts

        pc_start = torch.from_numpy(pc_start).float()
        pc_middle = torch.from_numpy(pc_middle).float()
        pc_end = torch.from_numpy(pc_end).float()
        occup_coords = torch.from_numpy(occup_coords).float()
        occup_labels = torch.from_numpy(occup_labels).float()

        inputs = {'point_cloud_1': pc_start, 'point_cloud_2': pc_middle, 'point_cloud_3': pc_end, 
                'coords_2': occup_coords, 'occup_labels_2': occup_labels,
               "center": center, "scale": scale, "obj_name": 'test'
        }

        return inputs
