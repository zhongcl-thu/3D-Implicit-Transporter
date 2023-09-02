import sys
import argparse
import os

sys.path.append("./")
import ipdb
import numpy as np
from core.utils.metric import compute_rep
from core.utils.viz import *
#from tools.eval_iou import nms_usip
from tqdm import trange


# adapted from usip
def nms_usip(keypoints_np, sigmas_np, NMS_radius, keypoints_np_2):
    '''
    :param keypoints_np: Mx3
    :param sigmas_np: M
    :return: valid_keypoints_np, valid_sigmas_np, valid_descriptors_np
    '''
    if NMS_radius < 0.01:
        return keypoints_np, sigmas_np

    valid_keypoint_counter = 0
    valid_keypoints_np = np.zeros(keypoints_np.shape, dtype=keypoints_np.dtype)
    valid_keypoints_np_2 = np.zeros(keypoints_np_2.shape, dtype=keypoints_np_2.dtype)
    valid_sigmas_np = np.zeros(sigmas_np.shape, dtype=sigmas_np.dtype)

    while keypoints_np.shape[0] > 0:

        max_idx = np.argmax(sigmas_np, axis=0)
        valid_keypoints_np[valid_keypoint_counter, :] = keypoints_np[max_idx, :]
        valid_keypoints_np_2[valid_keypoint_counter, :] = keypoints_np_2[max_idx, :]

        valid_sigmas_np[valid_keypoint_counter] = sigmas_np[max_idx]
        # remove the rows that within a certain radius of the selected minimum
        distance_array = np.linalg.norm(
            (valid_keypoints_np[valid_keypoint_counter:valid_keypoint_counter + 1, :] - keypoints_np), axis=1,
            keepdims=False)  # M
        mask = distance_array > NMS_radius  # M

        keypoints_np = keypoints_np[mask, ...]
        sigmas_np = sigmas_np[mask]

        keypoints_np_2 = keypoints_np_2[mask, ...]

        # increase counter
        valid_keypoint_counter += 1

    return valid_keypoints_np[0:valid_keypoint_counter, :], \
           valid_sigmas_np[0:valid_keypoint_counter], valid_keypoints_np_2[0:valid_keypoint_counter, :]


def build_correspondence(source_desc, target_desc):
    """
    Find the mutually closest point pairs in feature space.
    source and target are descriptor for 2 point cloud key points. [5000, 32]
    """
    distance = np.sqrt(2 - 2 * (source_desc @ target_desc.T))
    source_idx = np.argmin(distance, axis=1)
    source_dis = np.min(distance, axis=1)
    target_idx = np.argmin(distance, axis=0)
    target_dis = np.min(distance, axis=0)

    result = []
    for i in range(len(source_idx)):
        if target_idx[source_idx[i]] == i:
            result.append([i, source_idx[i]])
        # elif source_dis[i]<0.5:
        #     result.append([i, source_idx[i]])
    return np.array(result)


def filter_seg(kpt=None, joint_id=None, pc=None, inlier_radius=0.05):
    
    inx = np.abs(255.0 * (pc[:, 3]) - joint_id - 1) < 1e-4
    moveble_points = pc[inx][:, :3]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(moveble_points)

    # ipdb.set_trace()
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    # # colors = np.zeros((pc.shape[0], 3))
    # # colors[:, 0] = 255.0 * (pc[:, 3]) - 1
    # # pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    moveble_points = np.array(cl.points)
    
    # ipdb.set_trace()
    if kpt is not None:
        kpt_xyz = kpt[:, :3]
        # kpt_score = kpt[:, 3:]
        kpt_xyz = np.expand_dims(kpt_xyz, 1).repeat(moveble_points.shape[0], 1)
        
        dist_matrix = np.linalg.norm((kpt_xyz - moveble_points), axis=2)
        
        dist = dist_matrix.min(1)
        
        valid_index = dist < inlier_radius

        # viz_pc_keypoint(moveble_points, kpt[valid_index][:10, :3], 0.01)
        return kpt[valid_index], valid_index, moveble_points
    else:
        return moveble_points


def cal_fpfh_feature(points, voxel_size=0.015):
    radius_feature = 5 * voxel_size
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(np.repeat(points[..., 3:], 3, 1))
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    dowmpcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                downpcd,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)) 
    
    return np.concatenate((np.asarray(downpcd.points), np.asarray(downpcd.colors)[:, 0][:, None]), 1), dowmpcd_fpfh.data.T


def resolve_pose(kp1, kp2):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(kp1)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(kp2)

    temp_corr = [[i,i] for i in range(kp1.shape[0])]
    frag_corr = o3d.utility.Vector2iVector(temp_corr)

    pred_trans = o3d.pipelines.registration.TransformationEstimationPointToPoint().compute_transformation( \
        source=pcd1, target=pcd2, corres=frag_corr)
    
    return pred_trans


def find_neareset_point(kp, pts):
    kp = np.expand_dims(kp, 1).repeat(pts.shape[0], 1)
    dist_matrix = np.linalg.norm((kp - pts), axis=2)
    
    return dist_matrix.argmin(1)
    

def normaliza(fea):
    return fea/np.expand_dims(np.linalg.norm(fea, 2, 1), 1)


def cal_add_dis(pred_pose, gt_pose, cls_ptsxyz):
    pred_pts = np.dot(cls_ptsxyz, pred_pose[:, :3].T) + pred_pose[:, 3]
    gt_pts = np.dot(cls_ptsxyz, gt_pose[:, :3].T) + gt_pose[:, 3]
    mean_dist = np.mean(np.linalg.norm(pred_pts - gt_pts, axis=-1))
    return mean_dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute repeatability")
    parser.add_argument("--dataset_root", default="data/bullet_multi_joint_test", type=str)
    parser.add_argument("--test_root", default="exp/bullet/all_cat/0909kps6_std_var_kettel0.1_pad0_input10000_aug_diflen10_withkettle_seed111/test_result/seen", type=str)
    parser.add_argument("--test_type", type=str, default='seen')
    parser.add_argument("--keypoint_num", default=6, type=int)
    parser.add_argument("--inlier_radius", default=0.1, type=int)
    parser.add_argument('--method', default='ours', type=str)
    parser.add_argument('--fpfh', action='store_true')

    parser.add_argument('--test_id', default=0, type=int)
    
    args = parser.parse_args()

    if args.test_type == 'seen':
        test_cat_name = ['Refrigerator', 'FoldingChair', 'Laptop', 'Stapler', 
                    'TrashCan', 'Microwave', 'Toilet', 'Window', 'StorageFurniture', 'Kettle'] #,
    elif args.test_type == 'unseen':
        test_cat_name = ['Box', 'Phone', 'Dishwasher', 'Safe', 'Oven', 'WashingMachine', 
                        'Table', 'KitchenPot', 'Door'] #'Bucket', 
    test_type = args.test_type
    dataset_root = args.dataset_root
    
    kpts = np.load(args.test_root + '/kpts.npy', allow_pickle=True)
    rep_cat = {}
    dist_cat = {}
    add_dis_cat = {}

    dist_all = []
    for cat in test_cat_name:
        # if cat != 'Dishwasher':
        #     continue
        rep = 0
        dist = 0
        add_dis = 0
        num = 0

        for i in trange(100):
            kpts_i_1 = kpts.item()[('kp1', '{}'.format(cat), '{}'.format(i))]
            kpts_i_2 = kpts.item()[('kp2', '{}'.format(cat), '{}'.format(i))]

            pc_start = np.load(os.path.join(dataset_root, cat, str(i), 'pc_start.npy'), allow_pickle=True)
            pc_end = np.load(os.path.join(dataset_root, cat, str(i), 'pc_end.npy'), allow_pickle=True)

            pose_start = np.load(os.path.join(dataset_root, cat, str(i), 'transform_start.npy'), allow_pickle=True)
            pose_end = np.load(os.path.join(dataset_root, cat, str(i), 'transform_end.npy'), allow_pickle=True)
            transform_mat = pose_end @ np.linalg.inv(pose_start)
            selected_joint = np.loadtxt(os.path.join(dataset_root, cat, str(i), 'select_joint_id.txt'))

            if args.method != 'ours':
                pc_start, pc_start_desc = cal_fpfh_feature(pc_start)
                pc_end, pc_end_desc = cal_fpfh_feature(pc_end)

                index1 = find_neareset_point(kpts_i_1[:, :3], pc_start[:, :3])
                index2 = find_neareset_point(kpts_i_2[:, :3], pc_end[:, :3])

                kp_i_1_object = pc_start[index1]
                kp_i_2_object = pc_end[index2]
                
                corr = build_correspondence(normaliza(pc_start_desc[index1]), normaliza(pc_end_desc[index2]))
                kp_i_1_matched = kp_i_1_object[corr[:, 0]]
                kp_i_2_matched = kp_i_2_object[corr[:, 1]]
                try:
                    kp_i_2_matched_masked, mask_index, moveble_points2 = filter_seg(kp_i_2_matched, selected_joint, pc_end)
                except:
                    ipdb.set_trace()

                if kp_i_2_matched_masked.shape[0] == 0:
                    rep += 0
                    dist += 0.5
                    add_dis += 0.5
                    continue

                kp_i_1_matched_masked = kp_i_1_matched[mask_index]

                kp_i_2_matched_masked, _, kp_i_1_matched_masked = nms_usip(kp_i_2_matched_masked[:, :3], 
                                                                            kp_i_2_matched_masked[:, 3],
                                                                            0.05,
                                                                            kp_i_1_matched_masked[:, :3])
                
                if kp_i_1_matched_masked.shape[0] > args.keypoint_num:
                    kp_i_1_matched_masked = kp_i_1_matched_masked[:args.keypoint_num]
                    kp_i_2_matched_masked = kp_i_2_matched_masked[:args.keypoint_num]
                
                if kp_i_1_matched_masked.shape[0] == 0:
                    rep += 0
                    dist += 0.5
                    add_dis += 0.5
                    continue
                    
                pred_transform = resolve_pose(kp_i_1_matched_masked, kp_i_2_matched_masked)
            
            else:
                kp_i_1_matched_masked = kpts_i_1[:, :3]
                kp_i_2_matched_masked = kpts_i_2[:, :3]

                pred_transform = resolve_pose(kp_i_1_matched_masked, kp_i_2_matched_masked)
                # print(pred_transform)
                # print(transform_mat.astype(np.float32))
            
            movable_points1 = filter_seg(kpt=None, joint_id=selected_joint, pc=pc_start)
            if movable_points1.shape[0] == 0:
                continue
            add_dis += cal_add_dis(pred_transform[:3], transform_mat[:3], movable_points1)
            num += 1
            # try:
            #     kpts_i_1 = filter_seg(kpts_i_1, selected_joint, pc_start)
            #     kpts_i_2 = filter_seg(kpts_i_2, selected_joint, pc_end)
            # except:
            #     print(cat, i)
            #     rep += 0.5
            
            # if cat == 'Stapler':
            #     ipdb.set_trace()
            #     viz_pc_keypoint(pc_start[:, :3], kp_i_1_matched_masked[:, :3], 0.02)
            #     viz_pc_keypoint(pc_end[:, :3], kp_i_2_matched_masked[:, :3], 0.02)
            
            kp_i_12_matched_masked = (transform_mat[:3, :3] @ kp_i_1_matched_masked.T).T + transform_mat[:3, 3]
            dist_matrix = np.linalg.norm((kp_i_2_matched_masked - kp_i_12_matched_masked), 2, 1)
            dist += dist_matrix.mean()
            
            dist_all.append(dist_matrix)
            repeatibility = np.sum(dist_matrix < args.inlier_radius) / dist_matrix.shape[0]
            rep += repeatibility
            print(dist_matrix.mean())

            # viz_pc_keypoint(pc_end[:, :3], kp_i_12_matched_masked[:, :3], 0.015)

            # kp_i_12_pred_matched_masked = (pred_transform[:3, :3] @ kp_i_1_matched_masked.T).T + pred_transform[:3, 3]
            # viz_pc_keypoint(pc_end[:, :3], kp_i_12_pred_matched_masked[:, :3], 0.015)

            # ipdb.set_trace()

            # if cat == 'Microwave':
            #     ipdb.set_trace()
            #     kpts_i_12 = (transform_mat[:3, :3]@kpts_i_1[:, :3].T).T + transform_mat[:3, 3]
            #     viz_pc_keypoint(pc_start[:, :3], kpts_i_1[:, :3], 0.01)

            # viz_pc_keypoint(pc_start[:, :3], kpts_i_1[:, :3], 0.01)
            # viz_pc_keypoint(pc_end[:, :3], kpts_i_2[:, :3], 0.01)
            # ipdb.set_trace()
            #     print(compute_rep(kpts_i_2[:, :3], kpts_i_1[:, :3], 
            #                     transform_mat[:3], inlier_radius=args.inlier_radius))
        
        rep_cat[cat] = rep/num
        dist_cat[cat] = dist/num
        add_dis_cat[cat] = add_dis/num

    with open(args.test_root + '/{}_{}_{}_result_{}.txt'.format(args.method, args.inlier_radius, test_type, args.test_id), 'w', encoding='utf-8') as f:
        f.write('--------------- rep:\n')
        avg =  0
        for cat in test_cat_name:
            f.write('{}----{:.3f} \n'.format(cat, rep_cat[cat]))
            avg += rep_cat[cat]
        f.write('Mean----{:.3f} \n'.format(avg/len(test_cat_name)))
        
        f.write('--------------- dist:\n')
        f.write('--------------- dist:\n')
        avg =  0
        for cat in test_cat_name:
            f.write('{}----{:.3f} \n'.format(cat, dist_cat[cat]))
            avg += dist_cat[cat]
        f.write('Mean----{:.3f} \n'.format(avg/len(test_cat_name)))
        
        f.write('--------------- dist:\n')
        f.write('--------------- add_dist:\n')
        avg =  0
        for cat in test_cat_name:
            f.write('{}----{:.3f} \n'.format(cat, add_dis_cat[cat]))
            avg += add_dis_cat[cat]
        f.write('Mean----{:.3f} \n'.format(avg/len(test_cat_name)))

        f.write('Repeatability:------------:\n')
        
        dist_all = np.concatenate(dist_all)
        for thr in [0.05, 0.1, 0.15, 0.2, 0.25]:
            repeatibility = np.sum(dist_all < thr) / dist_all.shape[0]

            f.write('{}----{:.3f} \n'.format(thr, repeatibility))
    
