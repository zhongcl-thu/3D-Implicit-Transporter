import os
import ipdb
import tqdm
import shutil
import numpy as np
import open3d as o3d
from easydict import EasyDict as edict
import cv2
from sklearn.metrics import pairwise_distances_argmin, pairwise_distances_argmin_min

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import core.nets as nets
from core.datasets.test_dataset import *
from core.utils.match import nms
from core.utils.viz import *
import core.losses as losses
from core.utils.procrustes import rigid_transform_3d, quaternion_to_axis_angle
from pytorch3d.transforms import matrix_to_quaternion


class Evaluator:
    def __init__(self, C):
        self.C = C
        self.config_common = edict(C.config["common"])
        self.config_test = edict(C.config["test"])
        self.data_config = edict(self.C.config['data_info'])

        tmp = edict()
        tmp.log = {}
        self.tmp = tmp
        self.mean = lambda lis: sum(lis) / len(lis)


    def initialize(self, args):
        self.device = args.device
        self.multi_gpu = args.multi_gpu
        self.local_rank = args.local_rank
        self.test_model_root = args.test_model_root
        self.nprocs = torch.cuda.device_count()
        self.args = args
        
        self.create_model()
        self.create_dataset()
        self.create_dataloader()

        name = self.data_config.test_type
        
        self.save_root = os.path.join(args.test_model_root, 'test_result', name)
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)
            # shutil.copy(args.test_model_root+'/config_test.yaml', self.save_root)

        if self.config_common.get('train_pose_only', False):
            self.train_pose_only = True
        else:
            self.train_pose_only = False


    def create_model(self):
        model_path = os.path.join("checkpoints", None)
        print(model_path)
        config_public = self.config_common.public_params

        self.model = nets.model_entry(self.config_common.net, config_public)
        if model_path != None:
            try:
                self.model.load_state_dict(torch.load(os.path.join(self.test_model_root, model_path)))
            except:
                checkpoint = torch.load(os.path.join(self.test_model_root, model_path), lambda a,b:a)
                self.model.load_state_dict({k[7:]:v for k,v in checkpoint.items()})
        
        self.model.to(self.device)
        if self.multi_gpu:
            self.model = self.set_model_ddp(self.model)


    def set_model_ddp(self, m):
        m = torch.nn.SyncBatchNorm.convert_sync_batchnorm(m)
        return torch.nn.parallel.DistributedDataParallel(m,
                                                      device_ids=[self.local_rank],
                                                      output_device=self.local_rank,
                                                      broadcast_buffers=False)
    

    def create_dataset(self,):
        self.kp_radius = 0.01
        if self.data_config.dataset_name == 'bullet':
            self.test_dataset = Articulated_obj_bullet_test(self.C.config)
        elif self.data_config.dataset_name == 'real':
            self.test_dataset = Articulated_obj_bullet_test_real(self.C.config, 'test')


    def create_dataloader(self):
        self.test_loader = DataLoader(
                                self.test_dataset,
                                batch_size=self.config_common.batch_size,
                                shuffle=False,
                                num_workers=self.config_common.workers,
                                sampler = None,
                                pin_memory=True,
                                drop_last=False
                            )


    def create_loss(self):
        config_loss = self.config_common.loss
        config_public = self.config_common.public_params
        loss_list = []

        for item in config_loss:
            loss_list.append(losses.loss_entry(config_loss[item], config_public))
        
        self.multiloss = losses.MultiLoss(loss_list)
        for l in self.multiloss.losses:
            l.to(self.device)


    def create_optimizer(self, parameters_to_train):
        '''
        optimization for refining coordinates of keypoint 
        '''
        config_optim = self.config_test.optim
        
        optim_method = config_optim.type
        if optim_method == 'Adam':
            self.optimizer = optim.Adam(parameters_to_train, 
                                    lr=config_optim.kwargs.lr)
        elif optim_method == 'SGD':
            nesterov = config_optim.get("nesterov", False)
            self.optimizer = torch.optim.SGD(
                parameters_to_train,
                config_optim.kwargs.base_lr,
                momentum=config_optim.kwargs.momentum,
                weight_decay=config_optim.kwargs.weight_decay,
                nesterov=nesterov,
            )
        else:
            raise ValueError("do not support {} optimizer".format(optim_method))


    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        if isinstance(self.model, dict):
            for m in self.model.values():
                m.eval()
        else:
            self.model.eval()


    def prepare_data(self):
        tmp = self.tmp
        for k, v in tmp.input_var.items():
            if not isinstance(v, list):
                tmp.input_var[k] = v.cuda(non_blocking=True)


    def save_kpts(self,):
        self.set_eval()
        tmp = self.tmp

        kpts = {}
        
        with torch.no_grad():
            for iteration, tmp.input_var in enumerate(tqdm.tqdm(self.test_loader)):
                self.prepare_data()
                # ipdb.set_trace()
                outputs = self.model(tmp.input_var['point_cloud_1'], 
                                    tmp.input_var['point_cloud_2'],
                                    ) #p=tmp.input_var['coords_2']
                
                for i in range(tmp.input_var['point_cloud_1'].shape[0]):
                    kps1_i = outputs['kps1'][i].cpu().numpy() / 2
                    kps2_i = outputs['kps2'][i].cpu().numpy() / 2

                    pc1_i = tmp.input_var['point_cloud_1'][i].cpu().numpy()
                    pc2_i = tmp.input_var['point_cloud_2'][i].cpu().numpy()

                    scale1_i = tmp.input_var['scale'][i].cpu().numpy()
                    center1_i = tmp.input_var['center'][i].cpu().numpy()

                    scale2_i = tmp.input_var['scale'][i].cpu().numpy()
                    center2_i = tmp.input_var['center'][i].cpu().numpy()
                    
                    pcd1_i = scale1_i * pc1_i + center1_i
                    pcd2_i = scale2_i * pc2_i + center2_i

                    kps1_i = scale1_i * kps1_i + center1_i
                    kps2_i = scale2_i * kps2_i + center2_i

                    cat_name = tmp.input_var['cat_name'][i]
                    idx = tmp.input_var['id'][i]

                    kpts[('kp1', '{}'.format(cat_name), '{}'.format(idx))] = kps1_i
                    kpts[('kp2', '{}'.format(cat_name), '{}'.format(idx))] = kps2_i

            np.save(self.save_root + '/kpts_{}.npy'.format(self.args.test_id), kpts)


    def show_occupancy(self,):
        '''show object shape by marching cube
        '''
        self.set_eval()
        tmp = self.tmp

        with torch.no_grad():
            for i, tmp.input_var in enumerate(tqdm.tqdm(self.test_loader)):
                if i > 2:
                    self.prepare_data()
                    
                    outputs = self.model(tmp.input_var['point_cloud_1'], 
                                        tmp.input_var['point_cloud_2'],
                                        p=tmp.input_var['coords_2'])

                    pcd_data = np.array(tmp.input_var['point_cloud_1'].cpu().numpy()[0], dtype=np.float64)
                    
                    pcd_data2 = np.array(tmp.input_var['point_cloud_2'].cpu().numpy()[0], dtype=np.float64)

                    # pcd_data_show = make_o3d_pcd(pcd_data)
                    
                    color_v = [float(int('e4', 16))/255.0, float(int('9f', 16))/255.0, float(int('5e', 16))/255.0]
                    # color_v = [float(int('65', 16))/255.0, float(int('a3', 16))/255.0, float(int('b7', 16))/255.0]
                    colors = np.zeros_like(pcd_data)
                    colors[:, 0] = color_v[0]
                    colors[:, 1] = color_v[1]
                    colors[:, 2] = color_v[2]
                    
                    # pcd_data_o3d = make_o3d_pcd(pcd_data2, colors=colors)
                    # custom_draw_geometry_with_key_callback([pcd_data_o3d], 
                    #                                      os.path.join(self.save_root, 'pcd_feature_{}_2.png'.format(i)))
                    
                    # o3d.visualization.draw_geometries([pcd_data_show], )

                    # pred_occ = outputs['occ_transp'].cpu().numpy()[0] #logits.
                    # ipdb.set_trace()
                    pred_occ = outputs['occ_ori'].cpu().numpy()[0] #logits.
                    pred_occ = np.log(pred_occ + 1e-8) - np.log(1 - pred_occ + 1e-8)
                    pred_occ = pred_occ.reshape(self.config_test.grid_kwargs.x_res, 
                                                self.config_test.grid_kwargs.y_res, 
                                                self.config_test.grid_kwargs.z_res
                                                )
                    if pred_occ.shape[2] < pred_occ.shape[0]:
                        pred_occ = np.concatenate((pred_occ, 
                                        -20 * np.ones((pred_occ.shape[0], pred_occ.shape[1], 
                                                    pred_occ.shape[0] - pred_occ.shape[2]))), 2)
                    mesh = extract_mesh(pred_occ, threshold=0.4)
                    inputs_path = os.path.join(self.save_root, '{}.ply'.format(i))
                    
                    mesh.show()
                    # mesh.export(inputs_path)
                    #ipdb.set_trace()
                    kps2_i = outputs['kps2'][0].cpu().numpy() / 2

                    # viz_ply_keypoint(mesh, kps2_i, radius=0.05, 
                    #                 save_path= os.path.join(self.save_root, '{}.png'.format(i)))
                   
                    
    def show_input_saliency(self,):
        '''show saliency of each input point
        '''
        self.set_eval()
        tmp = self.tmp

        with torch.no_grad():
            for i, tmp.input_var in enumerate(tqdm.tqdm(self.test_loader)):
                self.prepare_data()

                outputs = self.model(tmp.input_var['point_cloud_1'], 
                                    tmp.input_var['occup_coords_1'], index='1')
                
                colors = np.zeros((outputs['sal1'][0].shape[0], 3))
                pred_sal = outputs['sal1'].cpu().numpy()[0]
                colors[:, 0] = pred_sal
                
                show_pcd = make_o3d_pcd(tmp.input_var['occup_coords_1'].cpu().numpy()[0],
                                        colors)
                
                o3d.visualization.draw_geometries([show_pcd]) #draw_geometries
           
