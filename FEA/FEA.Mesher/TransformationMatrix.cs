using System;

namespace FEA.Mesher.IGES
{
    public class TransformationMatrix
    {
        // Entity Type 124
        public TransformationMatrix(double[] Parameters)
        {
            T = new double[3];
            R = new double[3,3];
            R[0, 0] = Parameters[1]; // Top Row
            R[0, 1] = Parameters[2];
            R[0, 2] = Parameters[3];

            T[0] = Parameters[4];

            R[1, 0] = Parameters[5];
            R[1, 1] = Parameters[6];
            R[1, 2] = Parameters[7];

            T[1] = Parameters[8];

            R[2, 0] = Parameters[9];
            R[2, 1] = Parameters[10];
            R[2, 2] = Parameters[11];

            T[2] = Parameters[12];

        }
        double[,] R; // Rotation Matrix
        double[] T; // Translation Vector
    }
}

