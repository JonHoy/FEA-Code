using System;

namespace FEA.Mesher.IGES
{
    // Reference IGES SPEC v6.0 1998-01-05 pdf pp 161
    public class Rational_BSpline_Curve
	{
		public Rational_BSpline_Curve(double[] Parameters)
		{
			K = (int)Parameters[1];
			M = (int)Parameters[2];
			PROP1 = (int)Parameters[3];
			PROP2 = (int)Parameters[4];
            PROP3 = (int)Parameters[5];
            PROP4 = (int)Parameters[6];
            int N = 1 + K - M;
            int A = N + 2 * M;
            T = new double[A + 1];
            Array.Copy(Parameters, 7, T, 0, T.Length);
            W = new double[K + 1];
            Array.Copy(Parameters, 8 + A, W, 0, W.Length);
            PTS = new double[(K + 1) * 3];
            Array.Copy(Parameters, 8 + A + K, PTS, 0, PTS.Length);
            V0 = Parameters[12 + A + 4 * K];
            V1 = Parameters[13 + A + 4 * K];
            XNORM = Parameters[14 + A + 4 * K];
            YNORM = Parameters[15 + A + 4 * K];
            ZNORM = Parameters[16 + A + 4 * K];
            // using the Cox-de Boor formula to calculate the basis functions
		}

		int K; // Upper index of sum.
		int M; // Degree of basis functions
		int PROP1; // 0 = nonplanar, 1 = planar
		int PROP2; // 0 = open curve, 1 = closed curve
		int PROP3; // 0 = rational, 1 = polynomial
		int PROP4; // 0 = nonperiodic, 1 = periodic
		double[] T; // knot sequence
		double[] W; // Weights
		double[] PTS; // control points
        double V0; // starting parameter value
        double V1; // ending parameter value 
        // unit normal (if curve is planar)
        double XNORM;
        double YNORM;
        double ZNORM;
		
        //[,] B; // basis functions
	};
}

