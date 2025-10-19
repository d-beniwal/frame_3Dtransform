import numpy as np
import os

# ------------------------------------------------------------
def check_collinearity(points, tolerance=1e-6):
    """
    Check if we have at least 3 non-collinear points.
    
    Args:
        points: numpy array of shape (N, 3) - points to check
        tolerance: float - numerical tolerance for zero cross product
        
    Returns:
        tuple: (is_collinear, non_collinear_indices, collinear_message)
    """

    n_points = len(points)
    
    # Simple approach: check if first 3 points are non-collinear
    # If not, try different combinations until we find 3 non-collinear points
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            for k in range(j + 1, n_points):
                # Get three points
                p1, p2, p3 = points[i], points[j], points[k]
                
                # Create vectors
                v1 = p2 - p1
                v2 = p3 - p1
                
                # Check if vectors are parallel using cross product
                cross_product = np.cross(v1, v2)
                cross_magnitude = np.linalg.norm(cross_product)
                
                if cross_magnitude >= tolerance:
                    # Found 3 non-collinear points!
                    is_collinear = False
                    non_collinear_indices = [i, j, k]
                    collinear_message = "Sufficient non-collinear points found"
                    return is_collinear, non_collinear_indices, collinear_message
    
    # If we get here, all combinations of 3 points are collinear
    is_collinear = True
    non_collinear_indices = []
    collinear_message = "Can't find 3 non-collinear points. Unique 3D transformation is not possible."
    return is_collinear, non_collinear_indices, collinear_message



# ------------------------------------------------------------
def compute_transformation_matrix(points_frame1, points_frame2):
    """
    Compute the transformation matrix that transforms points from frame1 to frame2.
    This function implements the Procrustes superimposition algorithm that uses Kabsch algorithm for rotation
    and centroid conincidence for translation.

    Args:
        points_frame1: numpy array of shape (N, 3) - points in frame 1
        points_frame2: numpy array of shape (N, 3) - corresponding points in frame 2

    Returns:
        trans_matrix: 4x4 transformation matrix in homogeneous coordinates.
                    This matrix represents the complete rigid body transformation:
                    Bottom row [0, 0, 0, 1] remains unchanged for homogeneous coordinates
                    Top-left 3x3 block is the rotation matrix
                    Top-right 3x1 block is the translation vector
                    Example:
                    [[r11, r12, r13, tx],
                    [r21, r22, r23, ty],
                    [r31, r32, r33, tz],
                    [0,   0,   0,   1]]

        rmse: root mean square error of the fit
            This is the root mean square error of the fit between the transformed points and the actual points in frame2
    """

    # Convert inputs to numpy arrays to ensure we have proper matrix operations
    p1 = np.array(points_frame1)
    p2 = np.array(points_frame2)

    # Verify that both point sets have the same number of points and dimensions
    if p1.shape != p2.shape:
        raise ValueError("Point sets must have the same shape")
    
    # Ensure we're working with 3D points (each point has x, y, z coordinates)
    if p1.shape[1] != 3:
        raise ValueError("Points must be 3D (shape: N x 3)")

    n_points = p1.shape[0]
    
    # We need at least 3 points, but we'll do more detailed checking below
    if n_points < 3:
        raise ValueError("Need at least 3 points for unique 3D transformation")

    # Check if frame 1 & frame 2 point sets have enough non-collinear points
    is_collinear_p1, non_collinear_indices_p1, collinear_message_p1 = check_collinearity(p1)
    is_collinear_p2, non_collinear_indices_p2, collinear_message_p2 = check_collinearity(p2)

    if is_collinear_p1 or is_collinear_p2:
        error_msg = "Insufficient non-collinear points for unique 3D transformation.\n"
        if is_collinear_p1:
            error_msg += f"Frame 1 issues:\n{collinear_message_p1}\n"
        if is_collinear_p2:
            error_msg += f"Frame 2 issues:\n{collinear_message_p2}\n"

        error_msg += f"\nConsider adding more points or removing collinear points."
        
        raise ValueError(error_msg)


    # =============================
    # Remove translation by centering the point clouds
    # Calculate the centroid (geometric center) of each point cloud
    centroid1 = np.mean(p1, axis=0)  # Shape: (3,) - [cx, cy, cz] for frame1
    centroid2 = np.mean(p2, axis=0)  # Shape: (3,) - [cx, cy, cz] for frame2
    
    # Center the point clouds by subtracting their respective centroids
    # This removes the translational component, leaving only rotational differences
    # After centering, both point clouds have their centroid at origin (0,0,0)
    p1_centered = p1 - centroid1  # Shape: (N, 3)
    p2_centered = p2 - centroid2  # Shape: (N, 3)
    

    # =============================
    # Compute the cross-covariance matrix
    # The cross-covariance matrix H captures the correlation between
    # corresponding points in the two centered point clouds
    H = p1_centered.T @ p2_centered  # Shape: (3, 3)


    # =============================
    # SINGULAR VALUE DECOMPOSITION (SVD)
    # Apply SVD to the cross-covariance matrix: H = U * S * V^T
    U, S, Vt = np.linalg.svd(H)

    # =============================
    # Compute the optimal rotation matrix
    # The optimal rotation matrix is given by: R = V * U^T
    # This comes from the theory of orthogonal Procrustes problem
    # The rotation matrix R minimizes ||P2_centered - R * P1_centered||^2
    rotation_matrix = Vt.T @ U.T


    # =============================
    # Check if we have a proper rotation matrix (determinant = +1)
    # If det(R) = -1, we have a reflection, not a rotation
    # This can happen when the point clouds have opposite handedness
    if np.linalg.det(rotation_matrix) < 0:
        # To fix this, we flip the sign of the last column of V
        # (equivalently, the last row of V^T) and recompute R
        # This ensures we get a proper rotation matrix with det(R) = +1
        Vt[-1, :] *= -1
        rotation_matrix = Vt.T @ U.T
        

    # =============================
    # Compute the translation vector
    # The translation aligns the centroids after rotation:
    translation = centroid2 - rotation_matrix @ centroid1  # Shape: (3,)
    
    
    # =============================
    # Create a 4x4 homogeneous transformation matrix for convenient use
    # This matrix represents the complete rigid body transformation:
    # [R  t]  where R is 3x3 rotation matrix, t is 3x1 translation vector
    # [0  1]  bottom row enables matrix multiplication with homogeneous coords
    trans_matrix = np.eye(4)              # Start with 4x4 identity matrix
    trans_matrix[:3, :3] = rotation_matrix  # Set top-left 3x3 block to rotation matrix
    trans_matrix[:3, 3] = translation      # Set top-right 3x1 block to translation vector
    # Bottom row [0, 0, 0, 1] remains unchanged for homogeneous coordinates


    # =============================
    # Compute the root mean square error of the fit
    p1_transformed = (rotation_matrix @ p1.T).T + translation
    squared_errors = np.sum((p1_transformed - p2)**2, axis=1)  # ||difference||^2 for each point
    rmse = np.sqrt(np.mean(squared_errors))                    # Root mean square error
    
    return trans_matrix, rmse



# ------------------------------------------------------------
def transform_points_with_trans_matrix(points, transformation_matrix):
    """
    Transform multiple points from frame1 to frame2 using a 4x4 homogeneous transformation matrix.
    
    Args:
        points: numpy array of shape (N, 3) - points in frame1
        transformation_matrix: 4x4 transformation matrix. See compute_transformation_matrix() for details.
    
    Returns:
        transformed_points: Nx3 array of points in the transformed frame
            Example:
            [[x1, y1, z1],
            [x2, y2, z2],
            [x3, y3, z3],
            ...]
    """

    points = np.array(points)
    n_points = points.shape[0]
    
    # Convert to homogeneous coordinates
    points_homogeneous = np.hstack([points, np.ones((n_points, 1))])
    
    # Transform
    transformed_homogeneous = (transformation_matrix @ points_homogeneous.T).T
    
    return transformed_homogeneous[:, :3]



# ------------------------------------------------------------
def transform_points_from_rot_shift(points, x_rot, y_rot, z_rot, x_shift, y_shift, z_shift):
    """
    Transform points with known rotation and translation.
    
    Args:
        points: Nx3 array of points
        x_rot, y_rot, z_rot: rotation angles in degrees for axes x, y, z
        x_shift, y_shift, z_shift: translation values for axes x, y, z

    Returns:
        transformed_points: Nx3 array of points in the transformed frame
            Example:
            [[x1, y1, z1],
            [x2, y2, z2],
            [x3, y3, z3],
            ...]
    """

    # Convert point to numpy array
    points = np.array(points)

    # Convert rotation angles to radians
    x_rot_rad = np.deg2rad(x_rot)
    y_rot_rad = np.deg2rad(y_rot)
    z_rot_rad = np.deg2rad(z_rot)
    
    # Individual rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x_rot_rad), -np.sin(x_rot_rad)],
                   [0, np.sin(x_rot_rad), np.cos(x_rot_rad)]])
    
    Ry = np.array([[np.cos(y_rot_rad), 0, np.sin(y_rot_rad)],
                   [0, 1, 0],
                   [-np.sin(y_rot_rad), 0, np.cos(y_rot_rad)]])
    
    Rz = np.array([[np.cos(z_rot_rad), -np.sin(z_rot_rad), 0],
                   [np.sin(z_rot_rad), np.cos(z_rot_rad), 0],
                   [0, 0, 1]])
    
    # Combined rotation matrix (Z-Y-X order)
    R = Rz @ Ry @ Rx
    
    # Translation vector
    translation = np.array([x_shift, y_shift, z_shift])
    
    # Apply transformation: rotate then translate
    transformed_points = (R @ points.T).T + translation
    
    return transformed_points



# ------------------------------------------------------------
def save_transformation_matrix(trans_matrix, filepath, save_binary=True, save_text=True):
    """
    Save transformation matrix as both .npy and .txt files.
    
    Args:
        trans_matrix: 4x4 transformation matrix
        filename_base: Base filename (without extension)
    """

    # Save as .npy file (binary format - exact precision)
    if save_binary:
        npy_filepath = filepath + '.npy'
        np.save(npy_filepath, trans_matrix)
        print(f"Saved binary format: {npy_filepath}")
    
    # Save as .txt file (human-readable format)
    if save_text:
        txt_filepath = filepath + '.txt'
        np.savetxt(txt_filepath, trans_matrix, fmt='%.8f', delimiter='\t')
        print(f"Saved text format: {txt_filepath}")

    return None


# ------------------------------------------------------------
def load_trans_matrix_from_file(filepath):
    """
    Load transformation matrix from .npy file or .txt file.
    
    Args:
        filepath: .npy or .txt filepath
        
    Returns:
        T: 4x4 transformation matrix numpy array
    """

    if filepath.endswith('.npy'):
        trans_matrix = np.load(filepath)
    elif filepath.endswith('.txt'):
        trans_matrix = np.loadtxt(filepath)
    else:
        raise ValueError("Invalid file extension. Must be .npy or .txt")
    
    print(f"Loaded transformation matrix from: {filepath}")
    
    return trans_matrix



# ------------------------------------------------------------
def parse_points_file(filepath):
    """
    Parse text file containing two point sets. Each point set is a set of 3D points.
    The file should contain two sections: [POINTS_SET1] and [POINTS_SET2].
    Each section should contain a list of 3D points in the format: [x, y, z].

    Args:
        filepath: Path to the input text file

    Returns:
        points_set1: Nx3 array of points in the first set
        points_set2: Nx3 array of points in the second set
    """
    
    points_set1 = []
    points_set2 = []
    current_section = None
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Check for section headers
            if line.upper() in ['POINTS_SET1', 'FRAME1', 'SET1']:
                current_section = 'set1'
                continue
            elif line.upper() in ['POINTS_SET2', 'FRAME2', 'SET2']:
                current_section = 'set2'
                continue
            
            # Parse point coordinates
            try:
                coords = [float(x) for x in line.split()]
                if len(coords) != 3:
                    raise ValueError(f"Line {line_num}: Expected 3 coordinates, got {len(coords)}")
                
                if current_section == 'set1':
                    points_set1.append(coords)
                elif current_section == 'set2':
                    points_set2.append(coords)
                else:
                    raise ValueError(f"Line {line_num}: Point found outside of SET1/SET2 section")
                    
            except ValueError as e:
                raise ValueError(f"Line {line_num}: Invalid coordinate format - {e}")
    
    # Convert to numpy arrays
    points_set1 = np.array(points_set1)
    points_set2 = np.array(points_set2)
    
    # Validate point sets
    if len(points_set1) == 0:
        raise ValueError("No points found in SET1")
    if len(points_set2) == 0:
        raise ValueError("No points found in SET2")
    if len(points_set1) != len(points_set2):
        raise ValueError(f"Point sets must have same length: SET1={len(points_set1)} points, SET2={len(points_set2)} points")
    
    return points_set1, points_set2