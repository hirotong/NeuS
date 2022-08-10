'''
Author: Jinguang Tong
'''

import torch

from easydict import EasyDict as edict
import numpy as np


class Pose():
    """
    A class of operation on camera poses (PyTorch tensors with shape [..., 3, 4])
    each [3,4] camera pose takes the form: [R|t], where R is world2cam, t is translation
    """

    def __call__(self, R=None, t=None):
        # construct a camera pose from the given R and/or t
        assert R is not None or t is not None
        if R is None:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
            R = torch.eye(3, device=t.device).repeat(*t.shape[:-1], 1, 1)
        elif t is None:
            if not isinstance(R, torch.Tensor):
                R = torch.tensor(R)
            t = torch.zeros(R.shape[:-1], device=R.device)
        else:
            if not isinstance(R, torch.Tensor):
                R = torch.tensor(R)
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
        assert R.shape[:-1] == t.shape and R.shape[-2:] == (3, 3)
        R = R.float()
        t = t.float()
        pose = torch.cat([R, t[..., None]], dim=-1)     # [...,3,4]
        assert pose.shape[-2:] == (3, 4)
        return pose

    def invert(self, pose, use_inverse=False):
        # invert a camera pose
        R, t = pose[..., :3], pose[..., 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1, -2)
        t_inv = (-R_inv@t)[..., 0]
        pose_inv = self(R=R_inv, t=t_inv)
        return pose_inv

    def compose(self, pose_list):
        # compose a sequence of poses together
        # pose_new = poesN ∙ ... ∙ pose2 ∙ pose1
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new, pose)
        return pose_new

    def compose_pair(self, pose_a, pose_b):
        # pose_new(x) = pose_b ∙ pose_a
        R_a, t_a = pose_a[..., :3], pose_a[..., 3:]
        R_b, t_b = pose_b[..., :3], pose_b[..., 3:]
        R_new = R_b @ R_a
        t_new = (R_b @ t_a + t_b)[..., 0]
        pose_new = self(R=R_new, t=t_new)
        return pose_new

    def distance(self, pose_a, pose_b):
        # distance = inv(pose_b) @ pose_a
        R_a, t_a = pose_a[..., :3], pose_a[..., 3:]
        R_b, t_b = pose_b[..., :3], pose_b[..., 3:]
        R_b_inv = R_b.transpose(-1, -2)
        t_b_inv = (-R_b_inv @ t_b)[..., 0]
        pose_b_inv = self(R=R_b_inv, t=t_b_inv)
        angle = torch.acos((torch.trace(R_b_inv @ R_a) - 1) / 2).rad2deg()
        return self.compose([pose_a, pose_b_inv]), angle


class Lie():
    '''
    Lie algebra for SO(3) and SE(3) operations in PyTorch
    '''

    def so3_to_SO3(self, w):  # [..., 3]
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        R = I + A*wx + B*wx@wx
        return R

    def SO3_to_so3(self, R, eps=1e-6):  # [...,3,3]
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        # ? Possibly cause the parameters to be nan.
        theta = ((trace - 1) / 2).clamp(-1+eps, 1-eps).acos_()[...,
                                                               None, None] % np.pi  # ln(R) will explode if theta==pi
        # print("theta: ", theta)
        lnR = 1/(2*self.taylor_A(theta)+eps)*(R-R.transpose(-2, -1))
        w0, w1, w2 = lnR[..., 2, 1], lnR[..., 0, 2], lnR[..., 1, 0]
        w = torch.stack([w0, w1, w2], dim=-1)
        return w

    def se3_to_SE3(self, wu: torch.Tensor):   # [...,6]
        w, u = wu.split([3, 3], dim=-1)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=wu.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        C = self.taylor_C(theta)
        R = I + A*wx + B*wx@wx
        V = I + B*wx + C*wx@wx
        Rt = torch.cat([R, (V@u[..., None])], dim=-1)
        return Rt

    def SE3_to_se3(self, Rt: torch.Tensor, eps=1e-8):     # [...,3,4]
        R, t = Rt.split([3, 1], dim=-1)
        w = self.SO3_to_so3(R)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=Rt.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        invV = I - 0.5 * wx + (1 - A / (2*B))/(theta**2 + eps) * (wx@wx)
        u = (invV@t)[..., 0]
        wu = torch.cat([w, u], dim=-1)
        return wu

    def skew_symmetric(self, w):
        w0, w1, w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack([torch.stack([O, -w2, w1], dim=-1),
                          torch.stack([w2, O, -w0], dim=-1),
                          torch.stack([-w1, w0, O], dim=-1)], dim=-2)
        return wx

    def taylor_A(self, x, nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            if i > 0:
                denom *= (2*i) * (2*i+1)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans

    def taylor_B(self, x, nth=10):
        # Taylor expansion of (1-cos(x))/x^2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+1) * (2*i+2)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans

    def taylor_C(self, x, nth=10):
        # Taylor expansion of (x-sin(x))/x^3
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+2) * (2*i+3)
            ans = ans + (-1)**i*x**(2*i)/denom
        return ans


class Quaternion():

    def q_to_R(self, q: torch.Tensor):
        # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        qa, qb, qc, qd = q.unbind(dim=-1)
        R = torch.stack([torch.stack([1-2*(qc**2+qd**2), 2*(qb*qc-qd*qa), 2*(qb*qd+qa*qc)], dim=-1),
                         torch.stack([2*(qb*qc+qa*qd), 1-2*(qb**2+qd**2), 2*(qc*qd-qa*qb)], dim=-1),
                         torch.stack([2*(qb*qd-qa*qc), 2*(qc*qd+qa*qb), 1-2*(qb**2+qc**2)], dim=-1)], dim=-2)
        return R

    def R_to_q(self, R: torch.Tensor, eps=1e-8):     # [B,3,3]
        # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        row0, row1, row2 = R.unbind(dim=-2)
        R00, R01, R02 = row0.unbind(dim=-1)
        R10, R11, R12 = row1.unbind(dim=-1)
        R20, R21, R22 = row2.unbind(dim=-1)
        t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        r = (1+t+eps).sqrt()
        qa = 0.5*r
        qb = (R21-R12).sign()*0.5*(1+R00-R11-R22+eps).sqrt()
        qc = (R02-R20).sign()*0.5*(1-R00+R11-R22+eps).sqrt()
        qd = (R10-R01).sign()*0.5*(1-R00-R11+R22+eps).sqrt()
        q = torch.stack([qa, qb, qc, qd], dim=-1)
        for i, qi in enumerate(q):
            if torch.isnan(qi).any():
                K = torch.stack([torch.stack([R00-R11-R22, R10+R01, R20+R02, R12-R21], dim=-1),
                                 torch.stack([R10+R01, R11-R00-R22, R21+R12, R20-R02], dim=-1),
                                 torch.stack([R20+R02, R21+R12, R22-R00-R11, R01-R10], dim=-1),
                                 torch.stack([R12-R21, R20-R02, R01-R10, R00+R11+R22], dim=-1)], dim=-2)/3.0
                K = K[i]
                eigval, eigvec = torch.linalg.eigh(K)
                V = eigvec[:, eigval.argmax()]
                q[i] = torch.stack([V[3], V[0], V[1], V[2]])
        return q

    def invert(self, q):
        qa, qb, qc, qd = q.unbind(dim=-1)
        norm = q.norm(dim=-1, keepdim=True)
        q_inv = torch.stack([qa, -qb, -qc, -qd], dim=-1) / norm**2
        return q_inv

    def product(self, q1, q2):  # [B,4]
        q1a, q1b, q1c, q1d = q1.unbind(dim=-1)
        q2a, q2b, q2c, q2d = q2.unbind(dim=-1)
        hamil_prod = torch.stack([q1a*q2a-q1b*q2b-q1c*q2c-q1d*q2d,
                                  q1a*q2b+q1b*q2a+q1c*q2d-q1d*q2c,
                                  q1a*q2c-q1b*q2d+q1c*q2a+q1d*q2b,
                                  q1a*q2d+q1b*q2c-q1c*q2b+q1d*q2a], dim=-1)
        return hamil_prod


def to_hom(x):
    # get homogeneous coordinates of the input
    x_hom = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
    return x_hom


def rotation_distance(R1, R2, eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1@R2.transpose(-2, -1)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    angle = ((trace-1)/2).clamp(-1+eps, 1-eps).acos_()  # numerical stability near -1/+1
    return angle




if __name__ == '__main__':
    q = torch.tensor([2.9670e-01,  8.5651e-01,  4.2232e-01, -4.7684e-07])
    R = Quaternion().q_to_R(q)
    se3 = torch.zeros((1, 6))
    print(Lie().se3_to_SE3(se3))