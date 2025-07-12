import torch

def quaternion_to_matrix(q):
    """
    Converts a quaternion to a rotation matrix.

    Args:
        q (torch.Tensor): Input quaternion tensor of shape (batch_size, 4).

    Returns:
        torch.Tensor: Rotation matrix tensor of shape (batch_size, 3, 3).
    """

    if __debug__:
        assert len(q.shape)==2, "Quaternion must have at least 2 dimensions"
        assert q.shape[1] == 4, "Quaternion must have 4 components"

    batch=q.shape[0]
    
    q = normalize_vector(q)
    
    qx = q[...,0].view(batch, 1)
    qy = q[...,1].view(batch, 1)
    qz = q[...,2].view(batch, 1)
    qw = q[...,3].view(batch, 1)

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    xw, yw, zw = qx * qw, qy * qw, qz * qw
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) 
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) 
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) 
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) 
    
    return matrix.contiguous()

def normalize_vector(v):
    """
    Normalize a vector.

    Args:
        v (torch.Tensor): The input vector to be normalized.

    Returns:
        torch.Tensor: The normalized vector.

    """
    device = v.device
    
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, torch.tensor([1e-8], device=device))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag

    return v

def matrix_to_quaternion(mat):
    """
    Converts rotation matrices to quaternions

    Parameters
    ----------
    mat : torch.Tensor (...,3,3) rotation matrices

    Returns
    -------
    quat : torch.Tensor (...,4) quaternions 
            (x,y,z,w). Note that the real part is the last element.

    """

    device = mat.device

    w = torch.maximum(torch.sqrt((1.0 + mat[...,0,0] + mat[...,1,1] + mat[...,2,2])) / 2.0, torch.tensor([1e-7],device=device))
    x = (mat[..., 2, 1] - mat[..., 1, 2]) / (4.0 * w)
    y = (mat[..., 0, 2] - mat[..., 2, 0]) / (4.0 * w)
    z = (mat[..., 1, 0] - mat[..., 0, 1]) / (4.0 * w)

    quat = torch.stack([x, y, z, w], dim=-1)  # stack along last dimension

    return quat