import helpers
import numpy as np

# ------------------------------------------------------------
def main():
    """
    Main function to check the code by:
    1) Simulating the transformation of points through known angles and shifts
    2) Calculating the transformation matrix using the Procrustes superimposition algorithm
    3) Comparing the performance of calculated transformation matrix for new points with the known angles and shifts
    """

    print("Randomly sampling 4 points in frame 1")
    points_frame1 = np.random.rand(4, 3)

    print("Randomly selecting rotations and translations of axes for frame 2")
    rot_x,rot_y,rot_z = np.random.uniform(-350, 350, 3)
    x_shift,y_shift,z_shift = np.random.uniform(-10, 10, 3)
    print("Rotation angles (X, Y, Z) degrees: ", rot_x, rot_y, rot_z)
    print("Translation values (X, Y, Z): ", x_shift, y_shift, z_shift)
    print("--------------------------------")

    print("Transforming points in frame 1 to frame 2 through known angles and shifts")
    print("Points in frame 1: \n", points_frame1)
    points_frame2 = helpers.transform_points_from_rot_shift(points_frame1, rot_x, rot_y, rot_z, x_shift, y_shift, z_shift)
    print("Points in frame 2: \n", points_frame2)
    print("--------------------------------")

    print("Calculating the transformation matrix using the Procrustes superimposition algorithm")
    trans_matrix, rmse = helpers.compute_transformation_matrix(points_frame1, points_frame2)
    print("Transformation matrix: \n", trans_matrix)
    print("Root mean square error of the fit: ", rmse)
    print("--------------------------------")

    print("Randomly selecting 5 points in frame 1 for testing")
    points_frame1_test = np.random.rand(5, 3)

    print("Transforming points in frame 1 for testing to frame 2 through known angles and shifts")
    points_frame2_test = helpers.transform_points_from_rot_shift(points_frame1_test, rot_x, rot_y, rot_z, x_shift, y_shift, z_shift)

    print("Transforming points in frame 1 for testing to frame 2 using the transformation matrix")
    points_frame2_calc = helpers.transform_points_with_trans_matrix(points_frame1_test, trans_matrix)

    print("\nPoints in frame 1 for testing: \n", points_frame1_test)
    print("\nPoints in frame 2 for testing: \n", points_frame2_test)
    print("\nPoints in frame 2 from calculated transformation matrix: \n", points_frame2_calc)
    rmse = np.sqrt(np.mean((points_frame2_test - points_frame2_calc)**2))
    print("\nRoot mean square error", rmse)

    if rmse > 1e-3:
        print(f"================================================================")
        print(f"WARNING: Root mean square error of the fit is greater than 1e-3")
        print(f"THERE COULD BE ISSUES IN IMPLEMENTATION OF THE CODE")
        print(f"================================================================")

    else:
        print(f"================================================================")
        print(f"SUCCESS: Root mean square error is very low")
        print(f"THE CODE IS IMPLEMENTED CORRECTLY")
        print(f"================================================================")

if __name__ == '__main__':
    main()