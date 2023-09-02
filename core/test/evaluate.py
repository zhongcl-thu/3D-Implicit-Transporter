import os
import ipdb
import tqdm
import numpy as np
from easydict import EasyDict as edict

import torch
from torch.utils.data import DataLoader

import core.nets as nets
from core.datasets.test_dataset import *
from core.utils.viz import *
import core.losses as losses


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


    def create_model(self):
        model_path = self.data_config.get('model_path', None)
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
        if self.data_config.dataset_name == 'partnet':
            self.test_dataset = Articulated_Obj_Syn_Test(self.C.config)
        elif self.data_config.dataset_name == 'partnet_real':
            self.test_dataset = Articulated_Real_Obj(self.C.config, 'val')
        else:
            raise NotImplementedError("ERROR: dataset not implemented!")


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
                outputs = self.model(tmp.input_var['point_cloud_1'], 
                                    tmp.input_var['point_cloud_2'],
                                    )
                
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
                    
                    # show kpts
                    # viz_pc_keypoint(pcd1_i, kps1_i, kp_radius=0.03)

                    cat_name = tmp.input_var['cat_name'][i]
                    idx = tmp.input_var['id'][i]

                    kpts[('kp_{}'.format(iteration))] = kps1_i
                    kpts[('kp1', '{}'.format(cat_name), '{}'.format(idx))] = kps1_i
                    kpts[('kp2', '{}'.format(cat_name), '{}'.format(idx))] = kps2_i

            np.save(self.save_root + '/kpts.npy', kpts)
            

    def show_occupancy(self,):
        '''show object shape by marching cube
        '''
        self.set_eval()
        tmp = self.tmp

        with torch.no_grad():
            for i, tmp.input_var in enumerate(tqdm.tqdm(self.test_loader)):
                self.prepare_data()
                
                outputs = self.model(tmp.input_var['point_cloud_1'], 
                                    tmp.input_var['point_cloud_2'],
                                    p=tmp.input_var['coords_2'])

                pcd_data = np.array(tmp.input_var['point_cloud_1'].cpu().numpy()[0], dtype=np.float64)
                
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
                mesh.show()
           
