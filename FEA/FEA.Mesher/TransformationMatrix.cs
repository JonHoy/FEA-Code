using System;
using ManagedCuda.VectorTypes;

namespace FEA.Mesher
{
    public class TransformationMatrix
    {
        // Entity Type 124
        public TransformationMatrix(double ThetaX, double ThetaY, double ThetaZ, double X, double Y, double Z){
            R = new double[3,3];
            T = new double[3];

            double cosw = Math.Cos(ThetaX);
            double cosj = Math.Cos(ThetaY);
            double cosk = Math.Cos(ThetaZ);

            double sinw = Math.Sin(ThetaX);
            double sinj = Math.Sin(ThetaY);
            double sink = Math.Sin(ThetaZ);

            R[0, 0] = cosj * cosk;
            R[0, 1] = cosw * sink + sinw * sinj * cosk;
            R[0, 2] = sinw * sink - cosw * sinj * cosk;
            R[1, 0] = -1.0 * cosj * sink;
            R[1, 1] = cosw * cosk - sinw * sinj * sink;
            R[1, 2] = sinw * cosk - cosw * sinj * sink;
            R[2, 0] = sinj;
            R[2, 1] = -1.0 * sinw * cosj;
            R[2, 2] = cosw * cosj;
        } // generates rotation matrix and translation vector based on input rotation and translation components 
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
        public IGES.IGSPoint Transform(IGES.IGSPoint P) {
            var Ans = new IGES.IGSPoint(T);
            Ans.x += P.x * R[0, 0] + P.y * R[0, 1] + P.z * R[0, 2];
            Ans.y += P.x * R[1, 0] + P.y * R[1, 1] + P.z * R[1, 2]; 
            Ans.x += P.x * R[2, 0] + P.y * R[2, 1] + P.z * R[2, 2];
            return Ans;
        }
        double[,] R; // Rotation Matrix
        double[] T; // Translation Vector
    }
}

