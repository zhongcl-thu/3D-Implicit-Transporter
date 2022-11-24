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
                    
                    # kp1 = kps1_i[0]
                    # kp2 = kp1 + axis_angle[0].cpu().numpy() * 0.3
                    # viz_pc_keypoint(pcd1_i, np.stack((kp1, kp2), 0), kp_radius=0.03)

                    # cat_name = tmp.input_var['cat_name'][i]
                    # idx = tmp.input_var['id'][i]

                    pcd1_i = tmp.input_var['point_cloud_1_ori'][i].cpu().numpy()
                    # pc2_i = tmp.input_var['point_cloud_1_ori'][i].cpu().numpy()
                    cat_name = tmp.input_var['obj_name'][i]
                    instance_name = tmp.input_var['instance_name'][i]
                    # color_v = [float(int('e4', 16))/255.0, float(int('9f', 16))/255.0, float(int('5e', 16))/255.0]
                    #color_v = [float(int('65', 16))/255.0, float(int('a3', 16))/255.0, float(int('b7', 16))/255.0]
                    if not os.path.exists('paper_show/ours/real_video/{}'.format(cat_name)):
                        os.makedirs('paper_show/ours/real_video/{}'.format(cat_name))
                    # viz_pc_keypoint(pcd1_i, kps1_i, 0.03, 
                    #                 save_path='paper_show/ours/real_video/{}/{}_kpt.png'.format(cat_name, instance_name)) #
                    
                    # img = cv2.imread(img_root.format(int(cat), int(index)))
                    img = tmp.input_var['rgb1'][i].cpu().numpy()
                    cam_k = tmp.input_var['cam_k'][i].cpu().numpy()
                    
                    kps1_i = kps1_i.T
                    proj_coord = (cam_k @ kps1_i / kps1_i[2, :]).T

                    palette = np.array(sns.color_palette("bright", 6))*255.0

                    #img = img[:,:,::-1]
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    for i in range(len(proj_coord)):
                        cv2.circle(img, (int(proj_coord[i][0]), int(proj_coord[i][1])), 10, 
                                    (int(palette[i, 2]), int(palette[i, 1]), int(palette[i, 0])), 
                                    -4, 0)
                        # cv2.imshow('img', I)
                        # cv2.waitKey(0)
                        # cv2.imwrite('sift_keypoints.jpg', I)
                        # cv2.destroyAllWindows()

                    # cv2.imshow('img', img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    cv2.imwrite('paper_show/ours/real_video/{}/{}_kpt_rgb.png'.format(cat_name, instance_name), img)

                    # viz_pc_keypoint(pcd2_i, kps2_i, 0.02)
                    # T12 = tmp.input_var['transform'].float()

                    # kpts[('kp1', '{}'.format(cat_name), '{}'.format(idx))] = kps1_i
                    # kpts[('kp2', '{}'.format(cat_name), '{}'.format(idx))] = kps2_i
                    #transform_mats[('kp2', '{}'.format(cat_name), '{}'.format(idx))] = transform_mat

            ipdb.set_trace()
            np.save(self.save_root + '/kpts_{}.npy'.format(self.args.test_id), kpts)
            #np.save(self.save_root + '/transform_mats.npy', transform_mats)


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
                    #ipdb.set_trace()
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
                    ipdb.set_trace()
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
    
    
    def show_saliency_field_slice(self, projection=1):
        self.set_eval()
        tmp = self.tmp

        with torch.no_grad():
            for i, tmp.input_var in enumerate(tqdm.tqdm(self.test_loader)):
                self.prepare_data()

                shift = make_3d_grid([-0.5, 0.5, 150],
                                    [-0.5, 0.5, 150],
                                    [-0.5, 0.5, 150])
                outputs = self.model(tmp.input_var['point_cloud_1'], 
                        torch.from_numpy(shift).unsqueeze(0).cuda(), index='1')
                sal1 = outputs['sal1'][0].cpu().numpy().reshape(150, 150, 150) #DHW

                # select one of [0, 1, 2] to choose a projection direction.
                sal1_proj = np.max(sal1, projection) 

                plt.axis('off')
                plt.imshow(sal1_proj, cmap='Reds')

                fig = plt.gcf()
                #fig.set_size_inches(7.0/3, 7.0/3) #dpi = 300, output = 700*700 pixels
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                plt.margins(0, 0)
                fig.savefig(os.path.join(self.save_root, '{}_smpl_field.png'.format(i)), 
                            format='png', transparent=True, dpi=300, pad_inches = 0)
                plt.show()
    

    def show_axis(self, projection=1):
        self.set_eval()
        tmp = self.tmp

        with torch.no_grad():
            for i, tmp.input_var in enumerate(tqdm.tqdm(self.test_loader)):
                self.prepare_data()

                outputs = self.model(tmp.input_var['point_cloud_1'], 
                            tmp.input_var['point_cloud_2'],
                            tmp.input_var['point_cloud_middle'],
                            tmp.input_var['coords_2'], 
                            'Laptop')
                
                ipdb.set_trace()
                l, details = self.multiloss(**outputs, **tmp.input_var)
                
                axis12 = outputs['axis12'][0].cpu().numpy()
                state12 = outputs['state12'][0].cpu().numpy()

                #if tmp.input_var['gt_joint'][0] == 1:
                print('angle:', state12 * 180.0 / np.pi)
                
                #else:                
                    #print('translation:', state12)
                
                kp1 = tmp.input_var['point_cloud_1'][0][0].cpu().numpy()
                kp2 = kp1 + axis12

                viz_pc_keypoint(tmp.input_var['point_cloud_1'][0].cpu().numpy(),
                                 np.stack((kp1, kp2), 0), kp_radius=0.03)
           
