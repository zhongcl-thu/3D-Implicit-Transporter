import numpy as np
import torch
import os

from core.utils.transform import *
from core.datasets.train_dataset import *
from core.utils.viz import *


class Articulated_obj_bullet_test(Articulated_obj_dataset_bullet):
    def __init__(self, config):

        self.config = config

        config_data = config["data_info"]
        config_public_params = config["common"]["public_params"]

        self.pcd_root = config_data["data_path"]
        self.config_public_params = config_public_params
        self.input_num = config_public_params["input_pcd_num"]
        self.on_occupancy_num = config_public_params["on_occupancy_num"]
        self.off_occupancy_num = config_public_params["off_occupancy_num"]

        self.test_obj_names = config_data['test_cat_name']

        self.grid_sample = config['test'].get('grid_sample', False)
        self.noise_std = self.config['test'].get('noise_std', 0)
        self.down_sample = self.config['test'].get('downsample', 1)
        self.padding = config_public_params['padding']
        self.bound = 0.5 * (self.padding + 1)

        self.filenames = []
        for obj_name in self.test_obj_names:
            name_copy = np.array([obj_name] * 100)
            index_list = np.arange(100)
            self.filenames.extend(np.stack((name_copy, index_list), 1))


    def get_input(self, index):
        # get pcd
        name, idx = self.filenames[index]

        data_start = np.load(os.path.join(self.pcd_root, name, idx, 'pc_start.npy'), allow_pickle=True)
        data_end = np.load(os.path.join(self.pcd_root, name, idx, 'pc_end.npy'), allow_pickle=True)
        pose_start = np.load(os.path.join(self.pcd_root, name, idx, 'transform_start.npy'), allow_pickle=True)
        pose_end = np.load(os.path.join(self.pcd_root, name, idx, 'transform_end.npy'), allow_pickle=True)

        transform_mat = pose_end @ np.linalg.inv(pose_start)

        joint_id = np.loadtxt(os.path.join(self.pcd_root, name, idx, 'select_joint_id.txt'))

        if data_start.shape[0] > 15000:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data_start[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(np.repeat(data_start[..., 3:], 3, 1))
            downpcd = pcd.voxel_down_sample(voxel_size=0.015)
            data_start = np.concatenate((np.asarray(downpcd.points), np.asarray(downpcd.colors)[:, 0][:, None]), 1)
        
        if data_end.shape[0] > 15000:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data_end[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(np.repeat(data_end[..., 3:], 3, 1))
            downpcd = pcd.voxel_down_sample(voxel_size=0.015)
            data_end = np.concatenate((np.asarray(downpcd.points), np.asarray(downpcd.colors)[:, 0][:, None]), 1)
        
        return data_start[:, :3], data_end[:, :3], data_start[:, 3], data_end[:, 3], \
                name, idx, transform_mat, joint_id


    def normalize_pcd(self, pcd_data):
        bound_max = pcd_data.max(0)
        bound_min = pcd_data.min(0)

        pcd_data += np.random.randn(pcd_data.shape[0],
                                    pcd_data.shape[1]) * self.noise_std

        center = (bound_min + bound_max) / 2
        scale = (bound_max - bound_min).max()
        # scale = scale * (1 + self.padding)

        pcd_data = (pcd_data - center) / scale

        return pcd_data, center, scale
    

    def prepare_occupancy_data(self, pcd_data):
        z_max = self.config_public_params.get('z_max', 0.5) * (1 + self.padding)
        z_min = self.config_public_params.get('z_min', -0.5) * (1 + self.padding)
        if self.grid_sample:
            coords = make_3d_grid([-self.bound, self.bound, self.config['test']['grid_kwargs']['x_res']],
                                [-self.bound, self.bound, self.config['test']['grid_kwargs']['y_res']],
                                [z_min, z_max, self.config['test']['grid_kwargs']['z_res']],
                                type='HWD')
            
        else:
            on_surface_coords = pcd_data
            off_surface_x = np.random.uniform(-0.5, 0.5, 
                                                size=(self.off_occupancy_num, 1))
            off_surface_y = np.random.uniform(-0.5, 0.5, 
                                                size=(self.off_occupancy_num, 1))
            off_surface_z  = np.random.uniform(z_min, z_max, 
                                                size=(self.off_occupancy_num, 1))
            off_surface_coords = np.concatenate((off_surface_x, 
                                                off_surface_y, 
                                                off_surface_z), 
                                                axis=1)
            coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)

        return coords


    def __getitem__(self, index):
        pc_start_ori, pc_end_ori, label_start, label_end, \
            cat_name, idx, transform_mat, joint_id = self.get_input(index)

        # sample
        pc_start = self.prepare_input_data(pc_start_ori, 1)
        pc_end = self.prepare_input_data(pc_end_ori, self.down_sample)

        # normalize
        # pc_start, center_start, scale_start = self.normalize_pcd(pc_start)
        # pc_end, center_end, scale_end = self.normalize_pcd(pc_end)
        bound_max = np.maximum(pc_start.max(0), pc_end.max(0))
        bound_min = np.minimum(pc_start.min(0), pc_end.min(0))
        center = (bound_min + bound_max) / 2
        scale = (bound_max - bound_min).max() #/ (1 + self.padding)

        pc_start = (pc_start - center) / scale
        pc_end = (pc_end - center) / scale
        pc_end_ori = (pc_end_ori - center) / scale

        # pc_end_ori, center_end_ori, scale_end_ori = self.normalize_pcd(pc_end_ori)

        # get occp coords and labels
        occup_coords = self.prepare_occupancy_data(pc_end_ori)

        pc_start = torch.from_numpy(pc_start).float()
        pc_end = torch.from_numpy(pc_end).float()
        occup_coords = torch.from_numpy(occup_coords).float()

        # TODO
        joint_label = np.ones(1)
        joint_type_label = torch.from_numpy(joint_label).float()
        
        inputs = {'point_cloud_1': pc_start, 'point_cloud_2': pc_end,
                #  'center_1': center_start, 'center_2': center_end, 
                #  'scale_1': scale_start, 'scale_2': scale_end,
                'center': center, 'scale': scale, 
                 'label_1': label_start, 'label_2': label_end,
                 'coords_2': occup_coords, 'cat_name': cat_name, 'id': idx, 
                 'transform': transform_mat, 'joint_id': joint_id, 
                 "gt_joint": joint_type_label} #  'point_cloud_middle': pc_end, 
        
        return inputs


class Articulated_obj_bullet_test_real(Articulated_obj_dataset_real):
    def get_input(self, index):
        # get pcd
        cat_name, instance_name, name_index = self.filenames[index].split(' ')
        name_index = int(name_index)

        cat_name2, instance_name2, name_index2 = self.filenames[23].split(' ')
        
        mask_object = self.mask_dict[cat_name]
        
        cam_pts1, rgb1 = self.preprocess(mask_object, cat_name, instance_name, show=True)
        cam_pts2, rgb2 = self.preprocess(mask_object, cat_name, instance_name2, show=False)
        
        return cam_pts1, cam_pts2, cat_name, instance_name, rgb1, rgb2
