#!/usr/bin/env python3
# Author: hiro.tong

import sys
sys.path.append('.')
from colmap_utils.colmap_wrapper import run_colmap
from typing import Union
from skimage.transform import resize
import colmap_utils.read_write_model as read_model
import torch
import imageio
import os
import json
import numpy as np


def trans_t(t): return torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 0],
]).float()


def rot_phi(phi): return torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]
]).float()


def rot_theta(theta): return torch.Tensor([
    [np.cos(theta), 0, -np.sin(theta), 0],
    [0, 1, 0, 0],
    [np.sin(theta), 0, np.cos(theta), 0],
    [0, 0, 0, 1]
]).float()


def to_homo(pose):
    homo = torch.tensor([0, 0, 0, 1], dtype=pose.dtype, device=pose.device)
    if pose.dim() == 2:
        return torch.cat([pose, homo.reshape(1, -1)], dim=0)
    else:
        return torch.cat([pose, homo.reshape(1, 1, -1).repeat(pose.shape[0], 1, 1)], dim=1)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def load_rgb(file_path):
    img = imageio.imread(file_path)
    img = np.array(img).astype(float)

    # convert pixel values to [0, 1]
    img /= 255.
    img = img.transpose(2, 0, 1)

    return img


def load_rgba(file_path):
    img = imageio.imread(file_path)
    img = np.array(img).astype(float)

    rgb = img[..., :3] / 255.
    # rgb = rgb.transpose(2, 0, 1)

    mask = (img[..., -1] > 127.5).astype(float)[..., np.newaxis]

    rgb = rgb * mask + (1-mask)

    return rgb, mask


def srgb2linear(image: Union[np.ndarray, torch.Tensor]):
    '''
    Convert image in sRGB color space to LinearRGB color space
    https://www.zhangxinxu.com/wordpress/2017/12/linear-rgb-srgb-js-convert/
    '''
    mask = (image <= 0.04045)
    if isinstance(image, np.ndarray):
        dst_image = np.zeros_like(image, dtype=np.float32)
        dst_image[mask] = image[mask] / 12.92
        dst_image[~mask] = np.power((image[~mask] + 0.055) / 1.055, 2.4)
    else:
        dst_image = torch.zeros_like(image, dtype=torch.float, device=image.device)
        dst_image[mask] = image[mask] / 12.92
        dst_image[~mask] = torch.pow((image[~mask] + 0.055) / 1.055, 2.4)
    return dst_image


def linear2srgb(image: Union[np.ndarray, torch.Tensor]):
    '''
    Convert image in LinearRGB color space to sRGB color space
    https://www.zhangxinxu.com/wordpress/2017/12/linear-rgb-srgb-js-convert/
    '''

    mask = (image <= 0.0031308)
    if isinstance(image, np.ndarray):
        dst_image = np.zeros_like(image, dtype=np.float32)
        dst_image[mask] = image[mask] * 12.92
        dst_image[~mask] = 1.055 * np.power(image[~mask], 1.0 / 2.4) - 0.055
    else:
        dst_image = torch.zeros_like(image, dtype=torch.float, device=image.device)
        dst_image[mask] = image[mask] * 12.92
        dst_image[~mask] = 1.055 * torch.pow(image[~mask], 1.0 / 2.4) - 0.055
    return dst_image


def load_colmap_data(basedir):

    camerasfile = os.path.join(basedir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)

    #
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print('Cameras', len(list_of_keys))

    h, w, f = cam.height, cam.width, cam.params[0]
    hwf = np.array([h, w, f]).reshape([3, 1])

    imagesfile = os.path.join(basedir, 'sparse/0/images.bin')
    imdata = read_model.read_image_binary(imagesfile)

    w2c_mats = []
    bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])

    names = [imdata[k].name for k in imdata]
    print('Images #', len(names))
    perm = np.argsort(names)
    names = np.sort(names)
    imageid2idx = {}
    for i, k in enumerate(imdata):
        # adjust image_ids of pts3d
        imageid2idx[k] = i+1
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3, 1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    w2c_mats = np.stack(w2c_mats, 0)[perm]
    c2w_mats = np.linalg.inv(w2c_mats)

    poses = c2w_mats[:, :3, :4].transpose([1, 2, 0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1, 1, poses.shape[-1]])], 1)
    # This is different from the original implementation
    # ! must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t] (this is currently used)
    # switch to [r, u, -t] from [r, -u, t]
    # poses = np.concatenate([poses[:, 0:1, :], -poses[:, 1:2, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)

    points3dfile = os.path.join(basedir, 'sparse/0/points3D.bin')
    pts3d = read_model.read_points3D_binary(points3dfile)
    
    for k in pts3d:
        idx = [imageid2idx[imgid] for imgid in pts3d[k].image_ids]
        idx = np.array(idx)
        pts3d[k] = pts3d[k]._replace(image_ids = idx)

    return poses, pts3d, perm, names


def load_cam_data(basedir):
    cam_file = os.path.join(basedir, 'transforms.json')
    with open(cam_file, 'r') as fp:
        meta = json.load(fp)
    camera_angle_x = meta['camera_angle_x']

    return camera_angle_x


def save_poses(basedir, poses, pts3d, perm, names, camera_angle_x):
    metas = {}
    metas['camera_angle_x'] = camera_angle_x
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            if len(cams) < ind - 1:
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            cams[ind-1] = 1
        vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print('Points', pts_arr.shape, 'Visibility', vis_arr.shape)

    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose(2, 0, 1) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr == 1]
    print('Depth stats', valid_z.min(), valid_z.max(), valid_z.mean())

    frames = []
    save_arr = []
    for i in perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis == 1]
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.5)
        print(i, close_depth, inf_depth)
        # * switch to [r, u, -t] from [-u, r, -t]
        transform_matrix = np.concatenate([poses[:, 1:2, i], -poses[:, 0:1, i], poses[:, 2:3, i], poses[:, 3:4, i]], 1)
        # transform_matrix = np.concatenate([poses[:, :4, i], np.array([[0, 0, 0, 1.]])], axis=0)
        
        # transform_matrix = np.concatenate([transform_matrix, np.array([[0, 0, 0, 1.]])], axis=0)
        file_path = names[i].split('.')[0]
        frames.append({
            'file_path': file_path,
            'transform_matrix': transform_matrix.tolist(),
            'rotation': 0.,
        })
        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
    save_arr = np.array(save_arr)

    np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)

    metas['frames'] = frames
    with open(os.path.join(basedir, 'transforms_colmap.json'), 'w') as fp:
        json.dump(metas, fp, indent=4)


def gen_poses(basedir, image_dir, match_type, factors=None):

    files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
    if os.path.exists(os.path.join(basedir, 'sparse/0')):
        files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
    else:
        files_had = []
    if not all([f in files_had for f in files_needed]):
        print('Need to run COLMAP')
        run_colmap(basedir, image_dir=image_dir, match_type=match_type)
    else:
        print('Don\'t need to run COLMAP')
    print('Post-colmap')

    poses, pts3d, perm, names = load_colmap_data(basedir)

    camera_angle_x = load_cam_data(basedir)

    save_poses(basedir, poses, pts3d, perm, names, camera_angle_x)

    if factors is not None:
        print('Factors: ', factors)

    print('Done with imgs2poses')

    return True


def minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, f'images_{r}')
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, f'images_{r[1]}x{r[0]}')
        if not os.path.exists(imgdir):
            needtoload = True

    import subprocess
    from shutil import copy

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = f'images_{r}'
            resizearg = f'{int(100/r)}'
        else:
            name = f'images_{r[1]}x{r[1]}'
            resizearg = f'{r[1]}x{r[1]}'
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        subprocess.check_output(f'cp {imgdir_orig}/* {imgdir}', shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['magick', '-resize', resizearg, '-format', 'png', f'*.{ext}'])
        print(args)
        os.chdir(imgdir)
        subprocess.check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            subprocess(f'rm {imgdir}/*.{ext}', shell=True)
            print('Removed duplicates')
        print('Done')


def normalize(x):
    return x / np.linalg.norm(x)


def spherify_poses(poses):

    def p34_to_44(p): return np.concatenate(
        [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1./rad
    poses_reset[:, :3, 3] *= sc
    rad *= sc

    poses_reset = np.concatenate([poses_reset[:, :3, :4], np.broadcast_to(
        poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)

    return poses_reset

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    """
    Camera pose average for front-parallel cameras
    """
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w




def load_data(basedir, factor=None, width=None, height=None, load_imgs=True):

    poses_arr = np.load(os.path.join(basedir, 'poses_boudns.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose((1, 2, 0))
    bds = poses_arr[:, -2:].transpose([1, 0])

    img0 = [os.path.join(basedir, )]


if __name__ == '__main__':
    gen_poses(basedir='/home/hiro/dataset/NeRF/data/nerf_synthetic/insect_asymmetric_1.2_0.8_0.5',
              image_dir='train', match_type='exhaustive_matcher')
