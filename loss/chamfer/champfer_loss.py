import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
cd = load(name="cd",
          sources=["./loss/chamfer/chamfer_distance.cpp",
                   "./loss/chamfer/chamfer_distance.cu"])

class ChamferDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int)

        if not xyz1.is_cuda:
            cd.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1.cuda()
            dist2 = dist2.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()
            cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        if not graddist1.is_cuda:
            cd.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        else:
            gradxyz1 = gradxyz1.cuda()
            gradxyz2 = gradxyz2.cuda()
            cd.backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)

        return gradxyz1, gradxyz2

def batch_pairwise_dist(pcs1, pcs2):
    return ChamferDistanceFunction.apply(pcs1, pcs2)

class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, pcs1, pcs2):
        """
        Args:
            xyz1: tensor with size of (B, N, 3)
            xyz2: tensor with size of (B, M, 3)
        """
        assert pcs1.shape[2] == pcs2.shape[2]
        dist1, dist2 = ChamferDistanceFunction.apply(pcs1, pcs2)  # (B, N), (B, M)
        return dist1, dist2 #(dist1 + dist2) / 2

def test_chamfer(cuda=True):
    import time
    torch.random.manual_seed(0)
    x = torch.rand(1, 2048, 3)
    y = torch.rand(1, 2048, 3)

    if cuda:
        x,y = x.cuda(),y.cuda()

    loss = ChamferLoss()
    t =time.time()
    l = loss(x,y)
    print('CUDA: %s\nTime: %0.9f\n' % (cuda, time.time() - t))
    # print('CUDA: %s\nLoss: %.9f\nTime: %0.9f\n' %(cuda,l.item(),time.time()-t))

if __name__ == '__main__':
    test_chamfer(cuda=True)