"""
Author: Jinguang Tong
Affliction: Australia National University, DATA61, Black Mountain
"""

import os, json
import imageio
import torch
import cv2
import trimesh
import sys

sys.path.append(".")
import camera
import numpy as np
import torch.nn.functional as F
from typing import Dict
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from tqdm import tqdm
from skimage import measure
from .data_utils import load_rgba, pose_spherical, to_homo


class Database(Dataset):
    def __init__(self) -> None:
        pass

    def gen_rays_at(slef):
        raise NotImplementedError


class Blender(Database):
    def __init__(
        self,
        data_dir,
        camera_model=None,
        half_res=False,
        mode="train",
        use_gt_pose=False,
        regenerate=False,
        resolution=128,
        device="cuda",
    ):
        super(Blender, self).__init__()

        self.case = data_dir.split(os.sep)[-2]
        assert mode.lower() in ["train", "valid", "test"]
        self.mode = mode
        self.use_gt_pose = use_gt_pose
        self.half_res = half_res
        self.root = os.path.expanduser(data_dir)
        self.device = torch.device(device=device)
        # self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.scale = torch.tensor([1.0])

        with open(os.path.join(self.root, "transforms_{}.json".format(mode)), "r") as fp:
            metas_aruco = json.load(fp)
        with open(os.path.join(self.root, "transforms_{}_gt.json".format(mode)), "r") as fp:
            metas_gt = json.load(fp)
        # with open(os.path.join(self.root, 'transforms_{}.json'.format(mode)), 'r') as fp:
        #     colmap = json.load(fp)
        self.metas_gt = metas_gt
        self.metas_aruco = metas_aruco
        self.imgs = None
        self.masks = None
        self.poses = None
        self.poses_aruco = None
        self.params = None
        self._load_data()
        self.poses_aruco = self._load_pose(metas_aruco)
        self.poses_gt = self._load_pose(metas_gt)
        self.n_images = len(self.imgs)
        self.bbox_path = os.path.join(".", "data", f"visual_hull_{self.case}.obj")
        self.regenerate = False  # not os.path.exists(self.bbox_path) or regenerate
        if self.regenerate:
            self._compute_visual_hull(resolution=resolution)

        if camera_model:
            self._load_camera_model(camera_model)

        self.render_poses = torch.stack(
            [pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0
        )
        self.indices = range(len(self.imgs))

        self.poses = self.poses_gt if use_gt_pose else self.poses_aruco
        self.pose_all = self.poses

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])

        self.object_bbox_min = object_bbox_min[:3]
        self.object_bbox_max = object_bbox_max[:3]


        print("Load data: End")

    def __len__(self):
        return len(self.imgs)

    def _load_pose(self, meta_data: Dict):
        poses = []
        frames = meta_data["frames"]
        frames = sorted(frames, key=lambda x: int(x["file_path"].split("/")[-1].split("_")[-1]))
        for frame in tqdm(frames):
            fname = os.path.join(self.root, "train", frame["file_path"] + ".png")
            poses.append(np.array(frame["transform_matrix"]))
        poses = (np.array(poses)).astype(np.float32)

        return torch.from_numpy(poses).to(self.device)

    def _load_data(self):
        imgs = []
        masks = []
        poses = []
        frames = self.metas_gt["frames"]
        frames = sorted(frames, key=lambda x: int(x["file_path"].split("/")[-1].split("_")[-1]))
        for frame in tqdm(frames):
            fname = os.path.join(self.root, frame["file_path"] + ".png")
            img, mask = load_rgba(fname)
            imgs.append(img)
            masks.append(mask)
            poses.append(np.array(frame["transform_matrix"]))
        imgs = np.stack(imgs, axis=0).astype(np.float32)
        masks = np.stack(masks, axis=0).astype(np.float32)
        poses = (np.array(poses)).astype(np.float32)

        H, W = imgs[0].shape[:2]
        camera_angle_x = self.metas_gt["camera_angle_x"]
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

        if self.half_res:
            H, W, focal = H // 2, W // 2, focal / 2.0

            imgs_half_res = np.zeros([imgs.shape[0], H, W, imgs.shape[-1]])
            masks_half_res = np.zeros([masks.shape[0], H, W, masks.shape[-1]])
            for i, img in enumerate(imgs):
                imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                masks_half_res[i] = cv2.resize(
                    masks[i], (W, H), interpolation=cv2.INTER_NEAREST
                ).reshape([H, W, 1])
            imgs = imgs_half_res
            masks = masks_half_res

        self.intrinsics = torch.Tensor([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])
        self.intrinsics_inv = torch.inverse(self.intrinsics)

        self.imgs = torch.from_numpy(imgs).to(self.device)
        self.masks = torch.from_numpy(masks).to(self.device)
        # self.poses = torch.from_numpy(poses).to(self.device)
        self.H = H
        self.W = W
        self.focal = focal
        # transform from right-up-backward to right-down-forward
        self.transform = torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    def _load_camera_model(self, camera_model):

        model = RefineModel("./bbox/asymmetric_box.obj", None, self.n_images)
        state_dict = torch.load(camera_model)
        model.load_state_dict(state_dict, strict=False)

        self.scale = model.scale.detach()
        colmap_poses = model.get_camera_poses().detach()

        # self.poses = colmap_poses.clone().to(self.device)
        self.colmap_poses = colmap_poses.clone().to(self.device)

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        xx, yy = torch.meshgrid(tx, ty)
        xx = xx.t()
        yy = yy.t()

        p = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)  # H, W, 3
        p = torch.matmul(self.intrinsics_inv[None, None], p[..., None])  # H, W, 3, 1
        # ? from right-down-forward to right-up-backward
        p = torch.matmul(self.transform[None, None], p).squeeze()  # H, W, 3
        rays_d  = F.normalize(p, p=2, dim=-1)
        rays_d = torch.matmul(
            self.poses[img_idx, None, None, :3, :3], rays_d[..., None]
        ).squeeze()  # H, W, 3
        rays_o = self.poses[img_idx, None, None, :3, 3].expand(rays_d.shape)
        return rays_o, rays_d

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        tx = torch.randint(low=0, high=self.W, size=[batch_size])
        ty = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.imgs[img_idx][(ty, tx)]  # batch_size, 3
        mask = self.masks[img_idx][(ty, tx)]  # batch_size, 1
        p = torch.stack([tx, ty, torch.ones_like(tx)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_inv[None], p[..., None])  # batch_size, 3, 1
        p = torch.matmul(self.transform[None], p)
        rays_d = F.normalize(p, p=2, dim=1)
        rays_d = torch.matmul(self.poses[img_idx, None, :3, :3], rays_d).squeeze()
        rays_o = self.poses[img_idx, None, :3, 3].expand(rays_d.shape)  # batch_size, 3

        return torch.cat([rays_o, rays_d, color, mask], dim=-1).float()  # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        pass

    def near_far_from_sphere(self, rays_o, rays_d):
        """
        Get the near and far intersections for each ray on the sphere.
        """
        # ?
        a = torch.sqrt(torch.sum(rays_d ** 2, dim=-1, keepdim=True))
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        # all half chord length < r
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level=1):
        img = (self.imgs[idx].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return cv2.resize(img, (self.W // resolution_level, self.H // resolution_level))

    def _compute_visual_hull(self, resolution=512):
        """
        Compute a visual hull from object sihlouette.
        """
        min_x, max_x, min_y, max_y, min_z, max_z = -2, 2, -2, 2, -2, 2
        yy, xx, zz = np.meshgrid(
            np.linspace(min_y, max_y, resolution),
            np.linspace(min_x, max_x, resolution),
            np.linspace(min_z, max_z, resolution),
        )

        coord = np.concatenate(
            [xx[..., None], yy[..., None], zz[..., None]], axis=-1
        )  # res x res x res x 3
        volume = -np.ones_like(xx)
        intrinsics = self.intrinsics.detach().cpu().numpy()

        for i, (mask, pose) in enumerate(zip(self.masks, self.poses)):
            print(f"Process {i+1}/{len(self.imgs)} images")
            mask = mask.detach().cpu().numpy()
            pose = pose.detach().cpu().numpy()
            rot = np.linalg.inv(pose[:3, :3])

            coord_cam = coord - pose[:3, 3].reshape(1, 1, 1, 3)
            x_cam = np.sum(rot[0, :3].reshape(1, 1, 1, 3) * coord_cam, axis=-1)
            y_cam = -np.sum(rot[1, :3].reshape(1, 1, 1, 3) * coord_cam, axis=-1)
            z_cam = -np.sum(rot[2, :3].reshape(1, 1, 1, 3) * coord_cam, axis=-1)
            assert np.all(z_cam > 0)

            coord_cam = np.concatenate(
                [x_cam[..., None], y_cam[..., None], z_cam[..., None]], axis=-1
            )
            x_img = np.sum(intrinsics[0, :3].reshape(1, 1, 1, 3) * coord_cam, axis=-1)
            y_img = np.sum(intrinsics[1, :3].reshape(1, 1, 1, 3) * coord_cam, axis=-1)
            z_img = np.sum(intrinsics[2, :3].reshape(1, 1, 1, 3) * coord_cam, axis=-1)

            x_img /= z_img
            y_img /= z_img

            x_ind = np.logical_and(x_img >= 0, x_img < self.W - 0.5)
            y_ind = np.logical_and(y_img >= 0, y_img < self.H - 0.5)
            im_ind = np.logical_and(x_ind, y_ind)

            x_img_id = np.round(x_img[im_ind]).astype(np.int32)
            y_img_id = np.round(y_img[im_ind]).astype(np.int32)

            mask_ind = mask[y_img_id, x_img_id].squeeze()

            volume_ind = im_ind.copy()
            volume_ind[im_ind == 1] = mask_ind

            volume[volume_ind == 0] = 1

            print(f"Occupied voxel {im_ind.sum() / resolution ** 3 * 100:.2f}%")

        vertex, faces, normals, _ = measure.marching_cubes_lewiner(volume, 0)

        axis_len = float(resolution - 1) / 2.0
        vertex = (vertex - axis_len) / axis_len * 2
        mesh = trimesh.Trimesh(vertices=vertex, faces=faces, vertex_normals=normals)
        mesh.export(self.bbox_path)

        return mesh

    def update_pose(self, se3_refine):
        assert len(self.poses) == se3_refine.weight.shape[0]
        pose_refine = camera.Lie().se3_to_SE3(se3_refine.weight)
        self.poses = camera.Pose().compose([pose_refine, self.poses_aruco])

    def get_camera_matrix(self, idx):
        pose = self.poses[idx]
        K = self.intrinsics
        pose = to_homo(pose)
        R = torch.inverse(pose)
        K_inverse = self.intrinsics_inv
        return (R, K, pose, K_inverse)

    def get_image_mask(self, idx):
        assert idx < len(self.masks)

        return self.masks[idx]

    def __getitem__(self, index):
        poses_gt = self.poses_gt[index]
        poses_aruco = self.poses_aruco[index]
        mask = self.masks[index]
        K = self.intrinsics

        return torch.LongTensor([index]), poses_gt, poses_aruco, K, mask


if __name__ == "__main__":
    dataset = Blender(
        data_dir="~/dataset/NeRF/data/nerf_synthetic/insect_asymmetric_1.2_0.8_0.5_4",
        camera_model="/home/hiro/Code/MyProject/refractive-neus/train_camera_logs/22-03-10_21:23:13/model.pth",
        mode="train",
    )
    from torch.utils.data import DataLoader

    init_poses = []
    for i in range(len(dataset)):
        index, pose, colmap_pose, _ = dataset[i]
        # print(pose, colmap_pose, sep='\n')
        init_pose = pose @ torch.linalg.inv(colmap_pose)
        init_pose = init_pose[:3]
        init_se = camera.Lie().SE3_to_se3(init_pose)
        print(init_se)
        init_poses.append(init_se.detach().cpu().numpy())
    init_poses = np.stack(init_poses, axis=0)
    print(np.mean(init_poses, axis=0))

    # for i in range(len(dataset)):
    #     index, pose, colmap_pose, _ = (torch.stack([dataset[i][k], dataset[i+1][k]], dim=0) for k in range(4))
    #     rel_pose = pose[1] @ torch.linalg.inv(pose[0])
    #     rel_colmap = colmap_pose[1] @ torch.linalg.inv(colmap_pose[0])
    #     diff, angle = camera.Pose().distance(rel_pose[:3], rel_colmap[:3])
    #     print(rel_pose, rel_colmap, sep='\n')
    #     # print(camera.Lie().SE3_to_se3(rel_pose[:3]))
    #     # print(camera.Lie().SE3_to_se3(rel_colmap[:3]))
    #     print(angle)

