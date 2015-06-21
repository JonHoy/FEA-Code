﻿using System;

namespace FEA.Mesher.IGES
{
    // Reference IGES SPEC v6.0 1998-01-05 pdf pp 166

    //In the Far Future: (Could NURBS be used to computationally describe waves of turbulence?) ie Isosurface as a function of time 
    public class Rational_BSpline_Surface
    {
        public Rational_BSpline_Surface(double[] Parameters)
        {
            K1 = (int)Parameters[1];
            K2 = (int)Parameters[2];
            M1 = (int)Parameters[3];
            M2 = (int)Parameters[4];
            PROP1 = (int)Parameters[5];
            PROP2 = (int)Parameters[6];
            PROP3 = (int)Parameters[7];
            PROP4 = (int)Parameters[8];
            PROP5 = (int)Parameters[9];
            int N1 = 1 + K1 - M1;
            int N2 = 1 + K2 - M2;
            int A = N1 + 2 * M1;
            int B = N2 + 2 * M2;
            int C = (1 + K1) * (1 + K2);
            S = new double[N1 + 2 * M1 + 1];
            Array.Copy(Parameters, 10, S, 0, S.Length);
            T = new double[N2 + 2 * M2 + 1];
            Array.Copy(Parameters, 11 + A, T, 0, T.Length);
            W = new double[(K1 + 1) * (K2 + 1)];
            X = new double[W.Length];
            Y = new double[X.Length];
            Z = new double[Y.Length];
            Array.Copy(Parameters, 12 + A + B, W, 0, W.Length);
            var PTS = new double[3 * X.Length];
            Array.Copy(Parameters, 12 + A + B + C, PTS, 0, PTS.Length);
            for (int i = 0; i < X.Length; i++)
            {
                X[i] = PTS[3 * i];
                Y[i] = PTS[3 * i + 1];
                Z[i] = PTS[3 * i + 2];
            }
            U0 = Parameters[12 + A + B + 4 * C];
            U1 = Parameters[13 + A + B + 4 * C];
            V0 = Parameters[14 + A + B + 4 * C];
            V1 = Parameters[15 + A + B + 4 * C];
            Bi = new BSpline_Basis_Function(S, K1);
            Bj = new BSpline_Basis_Function(T, K2);
        }
        int K1; // Upper index of first sum. See Appendix B
        int K2; // Upper index of second sum. See Appendix B
        int M1; // Degree of first set of basis functions
        int M2; // Degree of second set of basis functions
        int PROP1; // 1 = Closed in first parametric variable direction
        int PROP2; // 1 = Closed in second parametric variable direction
        int PROP3; // 0 = Rational, 1 = Polynomial
        int PROP4; // 1 = Periodic in first parametric variable direction
        int PROP5; // 1 = Periodic in second parametric variable direction
        double[] S; // first knot sequence
        double[] T; // second knot sequence
        double[] W; // weights
        double[] X; // X Control Points
        double[] Y; // X Control Points
        double[] Z; // X Control Points
        double U0; // Starting value for first parametric direction
        double U1; // Ending value for first parametric direction
        double V0; // Starting value for second parametric direction
        double V1; // Ending value for second parametric direction
        BSpline_Basis_Function Bi; // B Spline Functions for first direction
        BSpline_Basis_Function Bj; // B Spline Functions for second direction
    }
}

