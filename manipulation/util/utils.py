import collections
import os
import queue
import shutil
import threading

import cv2
import dominate
import imageio
import numpy as np


def get_tableau_palette():
    """Get Tableau color palette (10 colors) https://www.tableau.com/.

    Returns:
        palette: 10x3 uint8 array of color values in range 0-255 (each row is a color)
    """
    palette = np.array([[ 78,121,167], # blue
                        [255, 87, 89], # red
                        [ 89,169, 79], # green
                        [242,142, 43], # orange
                        [237,201, 72], # yellow
                        [176,122,161], # purple
                        [255,157,167], # pink 
                        [118,183,178], # cyan
                        [156,117, 95], # brown
                        [186,176,172]  # gray
                        ],dtype=np.uint8)
    return palette


def transform_pointcloud(xyz_pts, rigid_transform):
    """Apply rigid transformation to 3D pointcloud.

    Args:
        xyz_pts: Nx3 float array of 3D points
        rigid_transform: 3x4 or 4x4 float array defining a rigid transformation (rotation and translation)

    Returns:
        xyz_pts: Nx3 float array of transformed 3D points
    """
    xyz_pts = np.dot(rigid_transform[:3,:3],xyz_pts.T) # apply rotation
    xyz_pts = xyz_pts+np.tile(rigid_transform[:3,3].reshape(3,1),(1,xyz_pts.shape[1])) # apply translation
    return xyz_pts.T


def get_heightmap(cam_pts, color_pts, segmentation_pts, view_bounds, heightmap_pix_sz, zero_level):
    """Get top-down (along z-axis) orthographic heightmap image from 3D pointcloud

    Args:
        cam_pts: Nx3 float array of 3D points in world coordinates
        color_pts: Nx3 uint8 array of color values in range 0-255 corresponding to cam_pts
        segmentation_pts: Nx1 int array of segmentation instance corresponding to cam_pts
        view_bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining region in 3D space of heightmap in world coordinates
        heightmap_pix_sz: float value defining size of each pixel in meters (determines heightmap resolution)
        zero_level: float value defining z coordinate of zero level (i.e. bottom) of heightmap
    
    Returns:
        depth_heightmap: HxW float array of height values (from zero level) in meters
        color_heightmap: HxWx3 uint8 array of backprojected color values in range 0-255 aligned with depth_heightmap
        segmentation_heightmap: HxW int array of segmentation instance aligned with depth_heightmap
    """

    heightmap_size = np.round(((view_bounds[1,1]-view_bounds[1,0])/heightmap_pix_sz,
                               (view_bounds[0,1]-view_bounds[0,0])/heightmap_pix_sz)).astype(int)

    # Remove points outside workspace bounds
    heightmap_valid_ind = np.logical_and(np.logical_and(
                          np.logical_and(np.logical_and(cam_pts[:,0] >= view_bounds[0,0],
                                                        cam_pts[:,0] <  view_bounds[0,1]),
                                                        cam_pts[:,1] >= view_bounds[1,0]),
                                                        cam_pts[:,1] <  view_bounds[1,1]),
                                                        cam_pts[:,2] <  view_bounds[2,1])
    cam_pts = cam_pts[heightmap_valid_ind]
    color_pts = color_pts[heightmap_valid_ind]
    segmentation_pts = segmentation_pts[heightmap_valid_ind]

    # Sort points by z value (works in tandem with array assignment to ensure heightmap uses points with highest z values)
    sort_z_ind = np.argsort(cam_pts[:,2])
    cam_pts = cam_pts[sort_z_ind]
    color_pts = color_pts[sort_z_ind]
    segmentation_pts = segmentation_pts[sort_z_ind]

    # Backproject 3D pointcloud onto heightmap
    heightmap_pix_x = np.floor((cam_pts[:,0]-view_bounds[0,0])/heightmap_pix_sz).astype(int)
    heightmap_pix_y = np.floor((cam_pts[:,1]-view_bounds[1,0])/heightmap_pix_sz).astype(int)

    # Get height values from z values minus zero level
    depth_heightmap = np.zeros(heightmap_size)
    depth_heightmap[heightmap_pix_y,heightmap_pix_x] = cam_pts[:,2]
    depth_heightmap = depth_heightmap-zero_level
    depth_heightmap[depth_heightmap < 0] = 0
    depth_heightmap[depth_heightmap == -zero_level] = 0

    # Map colors
    color_heightmap = np.zeros((heightmap_size[0],heightmap_size[1],3),dtype=np.uint8)
    for c in range(3):
        color_heightmap[heightmap_pix_y,heightmap_pix_x,c] = color_pts[:,c]
    
    # Map segmentations
    segmentation_heightmap = np.zeros((heightmap_size[0],heightmap_size[1]),dtype=np.int)
    segmentation_heightmap[heightmap_pix_y,heightmap_pix_x] = segmentation_pts[:, 0]

    return color_heightmap, depth_heightmap, segmentation_heightmap


def get_pointcloud(depth_img, color_img, segmentation_img, cam_intr, cam_pose=None):
    """Get 3D pointcloud from depth image.
    
    Args:
        depth_img: HxW float array of depth values in meters aligned with color_img
        color_img: HxWx3 uint8 array of color image
        segmentation_img: HxW int array of segmentation image
        cam_intr: 3x3 float array of camera intrinsic parameters
        cam_pose: (optional) 3x4 float array of camera pose matrix
        
    Returns:
        cam_pts: Nx3 float array of 3D points in camera/world coordinates
        color_pts: Nx3 uint8 array of color points
        color_pts: Nx1 int array of color points
    """

    img_h = depth_img.shape[0]
    img_w = depth_img.shape[1]

    # Project depth into 3D pointcloud in camera coordinates
    pixel_x,pixel_y = np.meshgrid(np.linspace(0,img_w-1,img_w),
                                  np.linspace(0,img_h-1,img_h))
    cam_pts_x = np.multiply(pixel_x-cam_intr[0,2],depth_img/cam_intr[0,0])
    cam_pts_y = np.multiply(pixel_y-cam_intr[1,2],depth_img/cam_intr[1,1])
    cam_pts_z = depth_img
    cam_pts = np.array([cam_pts_x,cam_pts_y,cam_pts_z]).transpose(1,2,0).reshape(-1,3)

    if cam_pose is not None:
        cam_pts = transform_pointcloud(cam_pts, cam_pose)
    color_pts = None if color_img is None else color_img.reshape(-1, 3)
    segmentation_pts = None if segmentation_img is None else segmentation_img.reshape(-1)

    return cam_pts, color_pts, segmentation_pts


def project_pts_to_2d(pts, camera_view_matrix, camera_intrisic):
    """Project points to 2D.

    Args:
        pts: Nx3 float array of 3D points in world coordinates.
        camera_view_matrix: 4x4 float array. A wrd2cam transformation defining camera's totation and translation.
        camera_intrisic: 3x3 float array. [ [f,0,0],[0,f,0],[0,0,1] ]. f is focal length.

    Returns:
        coord_2d: Nx3 float array of 2D pixel. (w, h, d) the last one is depth
    """
    pts_c = transform_pointcloud(pts, camera_view_matrix[0:3, :])
    rot_algix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]])
    pts_c = transform_pointcloud(pts_c, rot_algix) # Nx3
    coord_2d = np.dot(camera_intrisic, pts_c.T) # 3xN
    coord_2d[0:2, :] = coord_2d[0:2, :] / np.tile(coord_2d[2, :], (2, 1))
    coord_2d[2, :] = pts_c[:, 2]
    coord_2d = np.array([coord_2d[1], coord_2d[0], coord_2d[2]])
    return coord_2d.T


def mkdir(path, clean=False):
    """Make directory.
    
    Args:
        path: path of the target directory
        clean: If there exist such directory, remove the original one or not
    """
    if clean and os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)
        

def imretype(im, dtype):
    """Image retype.
    
    Args:
        im: original image. dtype support: float, float16, float32, float64, uint8, uint16
        dtype: target dtype. dtype support: float, float16, float32, float64, uint8, uint16
    
    Returns:
        image of new dtype
    """
    im = np.array(im)

    if im.dtype in ['float', 'float16', 'float32', 'float64']:
        im = im.astype(np.float)
    elif im.dtype == 'uint8':
        im = im.astype(np.float) / 255.
    elif im.dtype == 'uint16':
        im = im.astype(np.float) / 65535.
    else:
        raise NotImplementedError('unsupported source dtype: {0}'.format(im.dtype))

    assert np.min(im) >= 0 and np.max(im) <= 1

    if dtype in ['float', 'float16', 'float32', 'float64']:
        im = im.astype(dtype)
    elif dtype == 'uint8':
        im = (im * 255.).astype(dtype)
    elif dtype == 'uint16':
        im = (im * 65535.).astype(dtype)
    else:
        raise NotImplementedError('unsupported target dtype: {0}'.format(dtype))

    return im


def imwrite(path, obj):
    """Save Image.
    
    Args:
        path: path to save the image. Suffix support: png or jpg or gif
        image: array or list of array(list of image --> save as gif). Shape support: WxHx3 or WxHx1 or 3xWxH or 1xWxH
    """
    if not isinstance(obj, (collections.Sequence, collections.UserList)):
        obj = [obj]
    writer = imageio.get_writer(path)
    for im in obj:
        im = imretype(im, dtype='uint8').squeeze()
        if len(im.shape) == 3 and im.shape[0] == 3:
            im = np.transpose(im, (1, 2, 0))
        writer.append_data(im)
    writer.close()


def multithreading_exec(num, q, fun, blocking=True):
    """Multi-threading Execution.
    
    Args:
        num: number of threadings
        q: queue of args
        fun: function to be executed
        blocking: blocking or not (default True)
    """
    class Worker(threading.Thread):
        def __init__(self, q, fun):
            super().__init__()
            self.q = q
            self.fun = fun
            self.start()

        def run(self):
            while True:
                try:
                    args = self.q.get(block=False)
                    self.fun(*args)
                    self.q.task_done()
                except queue.Empty:
                    break
    thread_list = [Worker(q, fun) for i in range(num)]
    if blocking:
        for t in thread_list:
            if t.is_alive():
                t.join()


def html_visualize(web_path, data, ids, cols, others=[], title='visualization', threading_num=10, clean=True, save_figure=True, html_file_name='index', group_ids=None):
    """Visualization in html.
    
    Args:
        web_path: string; directory to save webpage. It will clear the old data!
        data: dict; 
            key: {id}_{col}. 
            value: figure or text
                - figure: ndarray --> .png or [ndarrays] --> .gif
                - text: string or [string]
        ids: [string]; name of each row
        cols: [string]; name of each column
        others: (optional) [dict]; other figures
            - name: string; name of the data, visualize using h2()
            - data: string or ndarray(image)
            - height: (optional) int; height of the image (default 256)
        title: (optional) string; title of the webpage (default 'visualization')
        threading_num: (optional) int; number of threadings for imwrite (default 10)
        clean: [bool] clean folder or not
        save_figure: [bool] save figure or not
        html_file_name: [str] html_file_name
        id_groups: list of (id_list, group_name)
    """
    mkdir(web_path, clean=clean)
    if save_figure:
        figure_path = os.path.join(web_path, 'figures')
        mkdir(figure_path, clean=clean)
        q = queue.Queue()
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                q.put((os.path.join(figure_path, key + '.png'), value))
            elif not isinstance(value, list) and isinstance(value[0], np.ndarray):
                q.put((os.path.join(figure_path, key + '.gif'), value))
        multithreading_exec(threading_num, q, imwrite)
    
    group_ids = group_ids if group_ids is not None else [('', ids)]

    with dominate.document(title=title) as web:
        dominate.tags.h1(title)
        for group_name, ids in group_ids:
            if group_name != '':
                dominate.tags.h2(group_name)
            with dominate.tags.table(border=1, style='table-layout: fixed;'):
                with dominate.tags.tr():
                    with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='center', width='64px'):
                        dominate.tags.p('id')
                    for col in cols:
                        with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='center'):
                            dominate.tags.p(col)
                for id in ids:
                    with dominate.tags.tr():
                        bgcolor = 'F1C073' if id.startswith('train') else 'C5F173'
                        with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='center', bgcolor=bgcolor):
                            for part in id.split('_'):
                                dominate.tags.p(part)
                        for col in cols:
                            with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='top'):
                                value = data[f'{id}_{col}']
                                if isinstance(value, str):
                                    dominate.tags.p(value)
                                elif isinstance(value, list) and isinstance(value[0], str):
                                    for v in value:
                                        dominate.tags.p(v)
                                else:
                                    dominate.tags.img(style='height:128px', src=os.path.join('figures', '{}_{}.png'.format(id, col)))
        for idx, other in enumerate(others):
            dominate.tags.h2(other['name'])
            if isinstance(other['data'], str):
                dominate.tags.p(other['data'])
            else:
                imwrite(os.path.join(figure_path, '_{}_{}.png'.format(idx, other['name'])), other['data'])
                dominate.tags.img(style='height:{}px'.format(other.get('height', 256)),
                    src=os.path.join('figures', '_{}_{}.png'.format(idx, other['name'])))
    with open(os.path.join(web_path, f'{html_file_name}.html'), 'w') as fp:
        fp.write(web.render())


def meshwrite(filename, verts, colors, faces=None):
    """Save 3D mesh to a polygon .ply file.

    Args:
        filename: string; path to mesh file. (suffix should be .ply)
        verts: [N, 3]. Coordinates of each vertex
        colors: [N, 3]. RGB or each vertex. (type: uint8)
        faces: (optional) [M, 4]
    """
    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    if faces is not None:
        ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write(
            "%f %f %f %d %d %d\n" %
            (verts[i, 0], verts[i, 1], verts[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]))

    # Write face list
    if faces is not None:
        for i in range(faces.shape[0]):
            ply_file.write("4 %d %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2], faces[i, 3]))

    ply_file.close()


def draw_action(image, position_start, position_end, cam_intrinsics, cam_view_matrix, thickness=3, tipLength=0.2, color=(186,176,172)):
    coord_3d = [position_start, position_end]
    coord_2d = project_pts_to_2d(np.array(coord_3d), cam_view_matrix, cam_intrinsics)
    p_start = (int(coord_2d[0, 1]), int(coord_2d[0, 0]))
    p_end = (int(coord_2d[1, 1]), int(coord_2d[1, 0]))

    image = cv2.arrowedLine(imretype(image, 'uint8'), p_start, p_end, color, thickness=thickness, tipLength=tipLength)
    return image


def angle(vec1, vec2, deg=False):
    _angle =  np.arccos(vec1.dot(vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2)) )
    if deg:
        _angle = np.rad2deg(_angle)
    return _angle