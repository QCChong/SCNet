import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.model_utils import calc_cd, calc_emd, calc_cd_single_side, MLP, fps, knnSearch, group_points

class LocalShape(nn.Module):
    def __init__(self, nPlanes, nShapes, npoints=None, k=16):
        super(LocalShape, self).__init__()
        self.k = k
        self.npoints = npoints
        self.mlp_planes = nn.Conv2d(3, nPlanes, (1,1), bias = False)
        self.mlp_shapes = MLP(channels=[nPlanes, nShapes])

    def forward(self, xyz):
        if self.npoints is not None:
            xyz_new = fps(xyz, self.npoints, BNC=True)                          # (B, npoints, 3), (B, npoints)
        else:
            xyz_new = xyz

        _, idx = knnSearch(xyz, xyz_new, self.k)                                # (B, npoints, K), (B, npoints, K)
        knn_points = group_points(xyz.transpose(2, 1).contiguous(), idx[:, :, 1:]) - \
                     xyz_new.transpose(2, 1).contiguous().unsqueeze(-1)         # (B, 3, npoints, k-1)

        knn_points_norm = torch.norm(knn_points, dim=1, keepdim=True) + 1e-8    # (B, 1, npoints, k-1)
        planes = self.mlp_planes(knn_points)/ knn_points_norm                   # (B, nPlanes, npoints, k-1)
        planes = torch.max(knn_points_norm * planes * torch.abs(planes), dim=-1)[0]        # (B, nPlanes, npoints)
        shapes = self.mlp_shapes(planes)                                        # (B, nShapes, npoints)
        return shapes, xyz_new, idx

class SA(nn.Module):
    def __init__(self, in_channels, out_channels, nPlanes, nShapes, npoints, k=16):
        super(SA, self).__init__()
        self.localShape = LocalShape(nPlanes, nShapes, npoints, k)
        self.mlp_f = MLP(channels=[in_channels + nShapes, out_channels], last_act=False)

    def forward(self, f, xyz):
        f_shapes, xyz_new, idx = self.localShape(xyz)

        f, _ = group_points(F.relu(f), idx).max(dim=-1)      # (B, C, npoints, k) -> (B, C, npoints)
        f = torch.cat([f, f_shapes], dim=1)
        f = self.mlp_f(f)
        return f, xyz_new

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.mlp = MLP(channels=[3, 32, 64], last_act=False)
        self.cov_sa1 = SA(in_channels=64, out_channels=128, nPlanes=64, nShapes=64, npoints=512, k=16)
        self.cov_sa2 = SA(in_channels=128, out_channels=256, nPlanes=64, nShapes=64, npoints=128, k=16)
        self.cov_sa3 = SA(in_channels=256, out_channels=1024, nPlanes=64, nShapes=64, npoints=32, k=16)

    def forward(self, x):
        f = self.mlp(x)
        xyz = x.transpose(2, 1).contiguous()                # (B, N, 3)

        f1, xyz_new = self.cov_sa1(f, xyz)                  # (B, 128, 512), (B, 512, 3)
        f2, xyz_new = self.cov_sa2(F.relu(f1), xyz_new)     # (B, 256, 128), (B, 128, 3)
        f3, xyz_new = self.cov_sa3(F.relu(f2), xyz_new)     # (B, 512, 32),  (B, 32, 3)
        g, _ = f3.max(dim=-1)
        return g

class Smooth(nn.Module):
    def __init__(self):
        super(Smooth, self).__init__()
        self.conv = MLP(channels=[3, 64, 64], last_act=False)
        self.linear = MLP(channels=[64+64, 64, 3])

    def forward(self, xyz):
        f = self.conv(xyz)

        g, _ = torch.max(f, dim=-1)
        f = torch.cat([f, g.unsqueeze(-1).repeat(1,1,f.shape[-1])], dim=1)
        xyz = xyz + self.linear(f)
        return xyz

class RE(nn.Module):
    def __init__(self, k=16, r=2):
        super(RE, self).__init__()
        self.k = k
        self.r = r
        self.mlp1 = MLP(channels=[3, 32, 64], last_act=True)
        self.mlp3 = MLP(channels=[64+64, 128, 3*self.r], last_act=False)
        self.attention = MLP(channels=[64, 128, 64], conv_type='2D', last_act=False)
        self.localShapes = LocalShape(nPlanes=64, nShapes=64, k=16)

    def forward(self, xyz):                                          # (B, 3, N)
        B, C, N = xyz.shape
        f_shapes, xyz_new, idx = self.localShapes(xyz.transpose(2, 1).contiguous())

        f = self.mlp1(xyz)                                           # (B, 64, N)
        f_knn = group_points(f, idx) - f.unsqueeze(-1)               # (B, C, N, k)
        w = self.attention(f_knn).softmax(dim=-1)
        f = (w * f_knn).sum(dim=-1)                                  # (B, 64, N)

        f = torch.cat([f, f_shapes], dim=1)                          # (B, 32+64, N)
        x = xyz.repeat(1,1,self.r) + 0.15*self.mlp3(f).view(B, -1, self.r*N)
        return x

class Decoder(nn.Module):
    def __init__(self, num_coarse=1024, num_dense=2048):
        super(Decoder, self).__init__()
        self.num_coarse = num_coarse
        self.num_dense = num_dense
        self.r = int(self.num_dense//self.num_coarse)
        self.linear = MLP(channels=[1024, 1024, 3 * self.num_coarse], conv_type='linear',  last_act=False)
        self.smooth = Smooth()
        self.up = RE(k=16, r=self.r)

    def forward(self, g, x):
        B, C = g.shape
        coarse = self.linear(g).view(B, 3, -1)           #(B, 3, 1024)

        xx_merge = torch.cat([x, coarse], dim=-1)        # (B, 3, 1024+2048)
        xx = fps(xx_merge, self.num_coarse, BNC=False)   # (B, 3, 1024)

        smooth = self.smooth(xx)
        out = self.up(smooth)                       # (B, 3, 2048)
        return coarse.transpose(2,1).contiguous(), smooth.transpose(2,1).contiguous(), out.transpose(2,1).contiguous()

class SCNet(pl.LightningModule):
    def __init__(self, num_coarse=1024, num_dense=2048, lrate=1e-4):
        self.save_hyperparameters()
        super(SCNet, self).__init__()
        self.lr = lrate
        self.alpha = 0.01
        self.num_coarse = num_coarse
        self.num_dense = num_dense
        self.E = Encoder()
        self.D = Decoder(num_coarse=num_coarse, num_dense=num_dense)

    def forward(self, x):
        x = x.transpose(2, 1).contiguous()
        return self.D(self.E(x), x)

    def share_step(self, batch):
        x, _, gt = batch                                       # (B,N,3)
        y_coarse, y_merge, y_detail = self.forward(x)          # (B,M,3), (B,2048,3)

        gt_coarse = fps(gt, y_coarse.shape[1], BNC=True)
        loss1, _ = calc_cd(y_coarse, gt_coarse)
        loss2, _ = calc_cd(y_merge, gt_coarse)
        loss3 = calc_emd(y_detail, gt)

        partial_matching = calc_cd_single_side(x, y_detail)
        rec_loss = loss1.mean() + loss2.mean() + 5*self.alpha*loss3.mean() + 0.5*self.alpha*partial_matching.mean()
        return  rec_loss

    def training_step(self, batch, batch_idx):
        return self.share_step(batch)

    def validation_step(self, batch, batch_idx):
        return self.share_step(batch)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.7)
        return [opt], [sched]

    def on_epoch_start(self):
        if self.current_epoch < 5:
            self.alpha = 0.01
        elif self.current_epoch < 15:
            self.alpha = 0.1
        elif self.current_epoch < 30:
            self.alpha = 0.5
        else:
            self.alpha = 1.0
