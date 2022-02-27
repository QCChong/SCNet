from utils.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
import torch
import torch.nn as nn

def farthest_point_sample(xyz, npoint):
    r"""
    Uses iterative furthest point sampling to select a set of npoint features that have the largest
    minimum distance

    Parameters
    ----------
    xyz : torch.Tensor
        (B, N, 3) tensor where N > npoint
    npoint : int32
        number of features in the sampled set

    Returns
    -------
    torch.Tensor
        (B, npoint) tensor containing the set
    """
    return pointnet2_utils.furthest_point_sample(xyz, npoint)

def gather_points(features, idx, tanspose = False):
    """
    Parameters
    ----------
    features : torch.Tensor
        (B, C, N) tensor

    idx : torch.Tensor
        (B, npoint) tensor of the features to gather

    Returns
    -------
    torch.Tensor
        (B, C, npoint) tensor
    """
    if tanspose:
        features = features.transpose(2, 1).contiguous()
        return pointnet2_utils.gather_operation(features, idx).transpose(2, 1).contiguous()
    else:
        return pointnet2_utils.gather_operation(features, idx)

def three_nn(unknown, known):
    r"""
        Find the three nearest neighbors of unknown in known
    Parameters
    ----------
    unknown : torch.Tensor
        (B, n, 3) tensor of known features
    known : torch.Tensor
        (B, m, 3) tensor of unknown features

    Returns
    -------
    dist : torch.Tensor
        (B, n, 3) l2 distance to the three nearest neighbors
    idx : torch.Tensor
        (B, n, 3) index of 3 nearest neighbors
    """
    return pointnet2_utils.three_nn(unknown, known)

def three_interpolate(features, idx, weight):
    # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
    r"""
        Performs weight linear interpolation on 3 features
    Parameters
    ----------
    features : torch.Tensor
        (B, c, m) Features descriptors to be interpolated from
    idx : torch.Tensor
        (B, n, 3) three nearest neighbors of the target features in features
    weight : torch.Tensor
        (B, n, 3) weights

    Returns
    -------
    torch.Tensor
        (B, c, n) tensor of the interpolated features
    """
    return pointnet2_utils.three_interpolate(features, idx, weight)

def group_points(features, idx, drop_fist=False):
    r"""

    Parameters
    ----------
    features : torch.Tensor
        (B, C, N) tensor of features to group
    idx : torch.Tensor
        (B, npoint, nsample) tensor containing the indicies of features to group with

    Returns
    -------
    torch.Tensor
        (B, C, npoint, nsample) tensor
    """
    if idx.dtype != torch.int32:
        idx = idx.int()
    f = pointnet2_utils.grouping_operation(features, idx.contiguous())

    if drop_fist:
        f = f[:,:,:,1:] - f[:,:,:,0].unsqueeze(-1)
        f[f==0] = 1e-10
    return f

def ball_query(radius, nsample, xyz, new_xyz):
    r"""

    Parameters
    ----------
    radius : float
        radius of the balls
    nsample : int
        maximum number of features in the balls
    xyz : torch.Tensor
        (B, N, 3) xyz coordinates of the features
    new_xyz : torch.Tensor
        (B, npoint, 3) centers of the ball query

    Returns
    -------
    torch.Tensor
        (B, npoint, nsample) tensor with the indicies of the features that form the query balls
    """
    return pointnet2_utils.ball_query(radius, nsample, xyz, new_xyz)

class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz, new_xyz, features=None):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """

        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = group_points(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = group_points(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz
        torch.cuda.empty_cache()
        return new_features


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features


if __name__ == '__main__':
    x = torch.rand(2,1024,3).cuda()
    idx = farthest_point_sample(x,512)
    y = gather_points(x.transpose(1,2).contiguous(),idx)


