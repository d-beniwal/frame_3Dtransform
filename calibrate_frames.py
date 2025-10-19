import argparse
import helpers

# ------------------------------------------------------------
def main():
    """
    Main function to calibrate frames.
    """

    parser = argparse.ArgumentParser(
        description='Compute transformation matrix from point correspondences',
        )

    parser.add_argument('refPointsFile',
                        type=str,
                        help='Path to the input text file containing reference points')

    parser.add_argument('-saveName',
                       default='trans_matrix',
                       help='Output filename base for the transformation matrix (default: trans_matrix)')

    args = parser.parse_args()

    print(f"\nReading reference point coordinates from: {args.refPointsFile}")
    ref_points_set1, ref_points_set2 = helpers.parse_points_file(args.refPointsFile)

    print(f"\nComputing transformation matrix...")
    trans_matrix, rmse = helpers.compute_transformation_matrix(ref_points_set1, ref_points_set2)

    print(f"\nSaving transformation matrix...")
    helpers.save_transformation_matrix(trans_matrix, args.saveName)

    print(f"\nTransformation matrix saved to: {args.saveName} .npy and .txt files")
    print(f"\nRoot mean square error of the fit: {rmse}")
    
    if rmse > 1e-3:
        print(f"================================================================")
        print(f"WARNING: Root mean square error of the fit is greater than 1e-3")
        print(f"CHECK THE REFERENCES POINTS ARE CORRECT AND REPEAT THE CALIBRATION")
        print(f"================================================================")

    print(f"\nDone!")

if __name__ == '__main__':
    main()