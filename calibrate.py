import numpy as np

def estimate_transform(camera_points, robot_points):
    """
    camera_points: Nx3 numpy array
    robot_points:  Nx3 numpy array

    Returns:
        R (3x3 rotation matrix)
        t (3-vector translation)
    """

    assert camera_points.shape == robot_points.shape
    assert camera_points.shape[1] == 3

    # Centroids
    centroid_cam = np.mean(camera_points, axis=0)
    centroid_robot = np.mean(robot_points, axis=0)

    # Center points
    cam_centered = camera_points - centroid_cam
    robot_centered = robot_points - centroid_robot

    # Covariance matrix
    H = cam_centered.T @ robot_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # Reflection correction
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_robot - R @ centroid_cam

    return R, t

points_in_camera = points_in_camera = np.array([[0.186, 0.252, 0.786],[0.095, 0.254, 0.756],[0.123, 0.085, 0.991],
    [0.160, 0.088, 1.003],[0.018, 0.215, 0.874],[0.035, 0.150, 0.927]], dtype=np.float32)


R, t = estimate_transform(points_in_camera, points_in_robot)
