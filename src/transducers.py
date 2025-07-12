import math
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def convert_transducer_to_global_coordinates(points_norm, positions, orientations, view_directions=None):
    """
    Converts sample points from the transducer coordinate frame to the global coordinate frame.

    parameters
    ---------
    points_norm : torch.Tensor (H, W, 2)

    positions : torch.Tensor (N, 3)
        X,Y,Z 

    orientations : torch.Tensor (N, 3, 3)
        Rotation matrix

    view_directions : torch.Tensor (H, 2)
        View direction for each scan line. If None, it won't be computed.

    returns
    -------
    dict containing:
        points : torch.Tensor (N, H, W, 3)
            rotated points

        view_directions : torch.Tensor (N, H, 3)
    """

    if __debug__:
        assert len(positions.shape) == 2, f"Positions must be of shape (N, 3). Got {positions.shape}"
        assert len(orientations.shape) == 3, f"Orientations must be of shape (N, 3, 3). Got {orientations.shape}"
        assert positions.shape[-1] == 3
        assert orientations.shape[0] == positions.shape[0], f"Number of positions must match number of orientations. Got {positions.shape} and {orientations.shape}"
        assert orientations.shape[-1] == 3 or orientations.shape[-2] == 3, f"Orientation matrix must be of shape (N, 3, 3). Got {orientations.shape}"
        assert points_norm.shape[-1] == 2, f"Points must be of shape (H, W, 2). Got {points_norm.shape}"
        assert len(points_norm.shape) == 3, f"Points must be of shape (H, W, 2). Got {points_norm.shape}"

        if view_directions is not None:
            assert len(view_directions.shape) == 2, f"View directions must be of shape (H, 2). Got {view_directions.shape}"
            assert view_directions.shape[-1] == 2, f"View directions must be of shape (H, 2). Got {view_directions.shape}"

    # points dim H,W,2. Add z=0
    ps = torch.cat([points_norm, torch.zeros((*points_norm.shape[:2], 1), device=points_norm.device)], dim=-1)

    # Repeat points to form a batch
    ps = ps[None].repeat(positions.shape[0], 1, 1, 1)
    
    # Reshape the points tensor to N x (H * W) x 3
    ps_flat = ps.reshape(ps.shape[0], -1, 3)

    # Perform batch matrix multiplication using torch.bmm
    ps_flat_rot = torch.bmm(orientations, ps_flat.transpose(1, 2)).transpose(1, 2)

    # Add the translation
    ps_flat_rot = ps_flat_rot + positions[None].permute(1, 0, 2)

    # Reshape the result back to N x H x W x 3
    ps_rot = ps_flat_rot.reshape_as(ps)

    out = {'points': ps_rot, 'view_directions': None}

    if view_directions is not None:
        # Add z=0
        vd = torch.cat([view_directions, torch.zeros((view_directions.shape[0], 1), device=view_directions.device)], dim=-1)[None] # 1xHx3
        vd = vd.repeat(positions.shape[0], 1, 1) # NxHx3
        vd = vd.transpose(1, 2) # Nx3xH
        view_directions_rot = torch.bmm(orientations, vd).transpose(1, 2) # NxHx3

        out['view_directions'] = view_directions_rot

    return out



def debug_assert(tensor, expected_shape, expected_type):
    """
    Asserts the shape and type of a given tensor.
    
    Args:
        tensor (torch.Tensor): The tensor to check.
        expected_shape (list): The expected shape of the tensor, where -1 allows for any size in that dimension.
        expected_type (type): The expected type of the tensor.
    
    Raises:
        AssertionError: If the tensor does not match the expected shape or type.
    """
    assert isinstance(tensor, expected_type), f"Input must be of type {expected_type.__name__}; got {type(tensor).__name__}"

    actual_shape = tensor.shape
    assert len(actual_shape) == len(expected_shape), f"Expected tensor of shape {expected_shape}, but got shape {actual_shape}"

    for actual, expected in zip(actual_shape, expected_shape):
        if expected != -1:
            assert actual == expected, f"Expected size {expected} at dimension, but got size {actual}"


def _render_baseline(planes, **kwargs):
    return {'planes_rendered': planes['positional'],
            'planes_clean': planes['positional'],
            'planes_raw': planes['positional']}




class Renderer(torch.nn.Module):

    def __init__(self, renderer_type):
        super(Renderer, self).__init__()

        self.renderer_type = renderer_type

    def forward(self, planes, **kwargs):
        
        if self.renderer_type == 'baseline':
            return _render_baseline(planes, **kwargs)
        else:
            raise NotImplementedError

def get_renderer(renderer_type):

    return Renderer(renderer_type=renderer_type)



class Transducer(torch.nn.Module):


    def __init__(self,
                 fov=0.2, # Horizontal field of view. meters
                 min_depth=0.03,
                 max_depth=0.2, # Maximum depth. meters
                 num_rays=100,
                 num_ray_points=100,
                 frequency=5, # MHz
                 resolution=1, # mm
                 tgc_factor=1, # db/cm/MHz
                 device='cpu',
                 **kwargs
                 ):
        

        super(Transducer, self).__init__()
        
        self.fov = fov
        self.num_ray_points = num_ray_points
        self.num_rays = num_rays
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.frequency = frequency
        self.resolution = resolution
        self.tgc_factor = tgc_factor

        self.subsample_factor = None
        self.device = device

        self.volume_postprocessor = None
        self.renderer_postprocessor = None


        self.renderer = get_renderer('baseline')

    def to(self, device, *args, **kwargs):
        super(Transducer, self).to(device, *args, **kwargs)

        self.device = device
        self.sample_points = self.sample_points.to(device)
        self.sample_points_norm = self.sample_points_norm.to(device)
        self.fov_vectors = self.fov_vectors.to(device)
        self.fov_mask = self.fov_mask.to(device)
        

        return self

    def set_renderer(self, renderer):

        self.renderer = get_renderer(renderer)
    

    # Shared functions
    def get_transducer_sample_points_global(self, positions, orientations):

        sample_points = self.sample_points[::self.subsample_factor, ::self.subsample_factor]

        return convert_transducer_to_global_coordinates(
            points_norm=sample_points,
            positions=positions,
            orientations=orientations,
            view_directions=self.fov_vectors[::self.subsample_factor],
        )
    
    def forward(self, volume, positions, orientations, time=None, **kwargs):
        """
        Args:
            volume (function or object): A function or object that takes a 4D torch.tensor of shape (N,H,W,2) representing
                                        normalized (-1,1) query points and a 1D torch.tensor of shape (N) representing the time.
                                        The function/object must return a 4D torch.tensor of shape (N,2,H,W) representing the plane.
            
            positions (torch.Tensor): A tensor of shape (N,3) representing the position of the transducer.

            orientations (torch.Tensor): A tensor of shape (N,3,3) representing the orientation of the transducer as a rotation matrix.

            time (torch.Tensor): A 1D tensor of shape (N) representing the time. If None, the time in volume function must be ignored.

        Returns:
            Output of the renderer for a given position, orientation, and time if applicable
        """

        if __debug__:
            debug_assert(positions, expected_shape=[-1, 3], expected_type=torch.Tensor)
            debug_assert(orientations, expected_shape=[-1, 3, 3], expected_type=torch.Tensor)
            if time is not None:
                debug_assert(time, expected_shape=[-1], expected_type=torch.Tensor)

        query_points = self.get_transducer_sample_points_global(positions=positions, orientations=orientations)
        
        volume_out = volume(x=query_points['points'],
                            time=time,
                            view_directions=query_points['view_directions'],
                            )
        
        # postprocess the volume output before rendering. E.g. warp planes in case of curvilinear transducer
        if self.volume_postprocessor is not None:
            volume_out = self.volume_postprocessor(volume_out, **kwargs)
        
        # Use the renderer to convert raw volume/encoder output to rendered planes
        renderer_out = self.renderer(volume_out, max_depth=self.max_depth-self.min_depth,
                                      resolution=self.subsample_factor)

        # postprocess the renderer output. E.g. warp planes in case of curvilinear transducer
        if self.renderer_postprocessor is not None:
            renderer_out = self.renderer_postprocessor(renderer_out, **kwargs)

        return renderer_out
    


#############################################################################################
##########################     LINEAR TRANSDUCER     ########################################
#############################################################################################
class LinearTransducer(Transducer):
    
    def __init__(self, **kwargs
                 ):

        super(LinearTransducer, self).__init__(**kwargs)

        self.type = 'linear'


        # Generate the field of view vectors (view direction)
        self.fov_vectors = torch.zeros(self.num_rays, 2, device=self.device)
        self.fov_vectors[:, 0] = 1   
        
        sp = self.generate_sample_points(fov=self.fov,
                                         min_depth=self.min_depth, 
                                         max_depth=self.max_depth, 
                                         num_rays=self.num_rays,
                                         num_ray_points=self.num_ray_points,
                                         ).permute(2, 0, 1)

        self.sample_points = sp
        
        # Normalize sample points to [-1,1].
        spn = (sp / sp.max()) * 2 - 1
        spn[1] += 1
        self.sample_points_norm = spn.permute(1, 2, 0).to(torch.float)
        self.sample_points = self.sample_points.permute(1, 2, 0).to(self.device)
        self.sample_points_norm = self.sample_points_norm.to(self.device)

        self.fov_mask = torch.ones(self.num_rays, self.num_ray_points, dtype=bool, device=self.device)

    def generate_sample_points(self, fov, min_depth, max_depth, num_rays, num_ray_points):
        y_range = torch.linspace(-fov/2, fov/2, num_rays)
        x_range = torch.linspace(min_depth, max_depth, num_ray_points)
        x_grid, y_grid = torch.meshgrid(x_range, y_range)
        points = torch.stack((x_grid.t(), y_grid.t()), dim=-1)
        return points
    





    

#############################################################################################
########################     CURVILINEAR TRANSDUCER    ######################################
#############################################################################################

def generate_fov_vectors(N, alpha):
    """
    Generates N 2D vectors spanning angle alpha
    
    Args:
        alpha (float): The angle, in degrees, which the vectors should span.
        
        N (int): The number of vectors to generate.
        
    Returns:
        torch.Tensor: A 2D tensor of shape (N, 2) representing the generated vectors.
    """
    
    # Convert angle to radians
    alpha_rad = math.radians(alpha)

    # Create the direction vector [1, 0] as a PyTorch tensor
    direction = torch.tensor([1.0, 0.0])

    # Calculate the step angle
    step_angle = alpha_rad / (N - 1)

    # Initialize the list to store the generated vectors
    vectors = []

    # Generate the vectors
    for i in range(N):
        # Calculate the current angle
        current_angle = -alpha_rad / 2 + i * step_angle

        # Calculate the rotation matrix
        rotation_matrix = torch.tensor([
            [math.cos(current_angle), -math.sin(current_angle)],
            [math.sin(current_angle), math.cos(current_angle)],
        ])

        # Apply the rotation matrix to the direction vector
        rotated_direction = torch.matmul(rotation_matrix, direction)
        
        # Append the vector to the list of generated vectors
        vectors.append(rotated_direction.unsqueeze(0))
    
    return torch.cat(vectors, dim=0)

def sample_points_along_vectors(P, V, N, min_dist, max_dist):
    """
    Samples N points along a list of vectors V from point P to the max distance D.
    
    Args:
        P (torch.Tensor): A 2D tensor of size (M, 2) representing the starting points of the vectors.
        
        V (torch.Tensor): A 2D tensor of size (M, 2) representing the directions of the vectors.
        
        N (int): An integer specifying the number of points to sample along each vector.
        
        min (float): A float specifying the minimum distance from the starting points.
        
        max_dist (float): A float specifying the maximum distance from the starting points.
        
    Returns:
        pos (torch.Tensor): A 3D tensor of size (M, N, 2) representing the positions of the N points along each vector.
    """
    # Normalize the vectors V
    V_norm = torch.nn.functional.normalize(V, dim=1)
    
    # Compute the step size between consecutive points
    step_size = (max_dist-min_dist) / (N - 1)
    
    # Generate the N points along each vector
    i = torch.arange(N, device=V.device).reshape(1, 1, -1)
    pos = P.unsqueeze(2) + i * step_size * V_norm.unsqueeze(2) + V_norm.unsqueeze(2)*min_dist
    
    return pos




class CurvilinearTransducer(Transducer):
    
    def __init__(self, **kwargs):

        super(CurvilinearTransducer, self).__init__(**kwargs)

        self.type = 'curvilinear'
        
        self.fov_vectors = generate_fov_vectors(self.num_rays, self.fov)
        sp = sample_points_along_vectors(torch.zeros((self.num_rays, 2)), 
                                         self.fov_vectors, 
                                         self.num_ray_points, 
                                         self.min_depth,
                                         self.max_depth).permute(1, 0, 2)
        
        self.sample_points = sp
        
        # Normalize sample points to [-1,1].
        spn = (sp / sp.max()) * 2 - 1
        spn[1] += 1
        spn[1] = spn[1] / spn[1].max() # Normalize y axis to [-1,1]
        self.sample_points_norm = spn.permute(1, 2, 0).to(torch.float)
        
        # Precompute inverse sampling grid used to warp annular sector to an image
        self.precompute_inverse_sampling_grid_threading(self.num_rays, self.num_ray_points)
        
        self.inverse_sample_points_norm = self.inverse_sample_points_norm.to(self.device)
        self.sample_points = self.sample_points.permute(1, 2, 0).to(self.device)
        self.sample_points_norm = self.sample_points_norm.to(self.device)
        self.fov_vectors = self.fov_vectors.to(self.device)


        self.fov_mask = (self.inverse_sample_points_norm.sum(-1) != -4)


        self.renderer_postprocessor = self.renderer_postprocessor_func

    def to(self, device, *args, **kwargs):
        super(CurvilinearTransducer, self).to(device, *args, **kwargs)
        self.inverse_sample_points_norm = self.inverse_sample_points_norm.to(device)

        return self

    def precompute_inverse_sampling_grid_worker(self, args):
        hh, ww, h, w, dist_th = args
        coord = torch.tensor([ww / w * 2 - 1, hh / h * 2 - 1])
        dists = torch.sqrt(((self.sample_points_norm - coord) ** 2).sum(-1))
        idx = torch.argmin(dists).item()
        i, j = divmod(idx, self.sample_points_norm.shape[1])
        dist = dists[i, j]
        if dist < dist_th:
            return hh, ww, j / w * 2 - 1, i / h * 2 - 1
        else:
            return hh, ww, -2, -2
    


    def precompute_inverse_sampling_grid_threading(self, h, w):
        grid = torch.ones((h, w, 2)) * -2
        dist_th = (2 / h + 2 / w) / 2

        loop_indices = [(hh, ww, h, w, dist_th) for hh in range(h) for ww in range(w)]

        with ThreadPoolExecutor() as executor:
            results = executor.map(self.precompute_inverse_sampling_grid_worker, loop_indices)

        for hh, ww, grid_x, grid_y in results:
            grid[hh, ww, 0] = grid_x
            grid[hh, ww, 1] = grid_y

        self.inverse_sample_points_norm = grid.to(torch.float)

    

    def convert_scan_to_polar_coordinates(self, scan_image):
        """
        Converts a scan-converted ultrasound image to polar coordinates.

        Args:
            scan_image (torch.Tensor): Input scan-converted image tensor of shape (batch_size, height, width, channels).

        Returns:
            torch.Tensor: Converted image in polar coordinates.
        """
        if __debug__:
            debug_assert(scan_image, expected_shape=[-1, -1, -1, -1], expected_type=torch.Tensor)

        if scan_image.shape[0] > 1:
            pts = self.sample_points_norm[::self.subsample_factor, ::self.subsample_factor][None].repeat(scan_image.shape[0], 1, 1, 1)
        else:
            pts = self.sample_points_norm[::self.subsample_factor, ::self.subsample_factor][None]

        polar_image = torch.nn.functional.grid_sample(scan_image.to(torch.float), pts, align_corners=True)

        return polar_image

    def convert_polar_to_scan_coordinates(self, polar_image):
        """
        Converts an ultrasound image in polar coordinates back to scan-converted format.

        Args:
            polar_image (torch.Tensor): Input polar image tensor of shape (batch_size, height, width, channels).

        Returns:
            torch.Tensor: Converted image in scan-converted format.
        """
        if __debug__:
            debug_assert(polar_image, expected_shape=[-1, -1, -1, -1], expected_type=torch.Tensor)

        if polar_image.shape[0] > 1:
            pts = self.inverse_sample_points_norm[::self.subsample_factor, ::self.subsample_factor][None].repeat(polar_image.shape[0], 1, 1, 1)
        else:
            pts = self.inverse_sample_points_norm[::self.subsample_factor, ::self.subsample_factor][None]

        scan_image = torch.nn.functional.grid_sample(polar_image.to(torch.float), pts, align_corners=True)

        return scan_image
    
        
    def renderer_postprocessor_func(self, planes, output_mode='annular_sector'):

        assert output_mode in ['annular_sector', 'plane'], f"output mode must be either 'annular_sector' or 'plane'. Got: {output_mode}"

        if output_mode == 'plane':
            for k in planes.keys():
                planes[k] = self.convert_polar_to_scan_coordinates(planes[k])
            return planes
        elif output_mode == 'annular_sector':
            return planes
        else:
            raise NotImplementedError(f"output mode: {output_mode} is not available")


    def batched_plane_to_annular_sector(self, data, batch_size=32, result_device='cpu', dtype=torch.float32):
        n = len(data)

        processed_data = torch.zeros((n, 1, self.num_rays, self.num_ray_points), device=result_device, dtype=dtype)

        with torch.no_grad():
        
            for i in tqdm(range(0, n, batch_size), desc="Converting images to annular sectors: "):
                batch = data[i:i + batch_size]
                batch = batch.to(self.device)

                if batch.dtype == torch.uint8:
                    batch = batch.float() / 255.0
                
                processed_batch = self.convert_scan_to_polar_coordinates(batch).detach().to(result_device)

                if dtype == torch.float32:
                    processed_data[i:i + batch_size] = processed_batch
                elif dtype == torch.float16:
                    processed_data[i:i + batch_size] = processed_batch.half()
                elif dtype == torch.uint8:
                    processed_data[i:i + batch_size] = (processed_batch * 255).byte()
                else:
                    raise ValueError(f"Unsupported dtype: {dtype}")

        
        return processed_data

def get_transducer(transducer_type='curvilinear', *args, **kwargs):

    if transducer_type == 'curvilinear':
        return CurvilinearTransducer(*args, **kwargs)
    elif transducer_type == 'linear':
        return LinearTransducer(*args, **kwargs)
    else:
        raise ValueError('Unknown transducer type: {}'.format(transducer_type))


