import numpy as np
import argparse
import helpers

# ------------------------------------------------------------
def main():
    """
    Main function to transform coordinates of a point cloud from one frame to another using a transformation matrix.
    """

    parser = argparse.ArgumentParser(
        description='Transform coordinates of a point cloud from one frame to another using a transformation matrix',
        )

    parser.add_argument('transMatFile',
                        type=str,
                        help='Path to the transformation matrix file (.npy or .txt)')

    parser.add_argument('-p',
                        type=str,
                        help="Single x,y,z coordinates of a point (e.g., '1.0,2.0,3.0')")

    args = parser.parse_args()
    
    #Convert the input arg -p to a numpy array of shape (1, 3)
    point = np.array([float(x) for x in args.p.split(',')]).reshape(1, 3)

    print(f"Reading transformation matrix from: {args.transMatFile}")
    trans_matrix = helpers.load_trans_matrix_from_file(args.transMatFile)
    transformed_point = helpers.transform_points_with_trans_matrix(point, trans_matrix)

    print(f"Original point: {point}")
    print(f"Transformed point: {transformed_point}")
    print(f"Done!")

if __name__ == '__main__':
    main()