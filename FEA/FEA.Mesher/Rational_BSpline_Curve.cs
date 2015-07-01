using System;
using ManagedCuda.VectorTypes;

namespace FEA.Mesher.IGES
{
    // Reference IGES SPEC v6.0 1998-01-05 pdf pp 161
    public class Rational_BSpline_Curve
	{
        public Rational_BSpline_Curve(double[] Parameters, TransformationMatrix _R = null, int _ParameterId = -1)
		{
            R = _R;
            ParameterId = _ParameterId;
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
            var PTS = new double[(K + 1) * 3];
            Array.Copy(Parameters, 8 + A + K, PTS, 0, PTS.Length);
            X = new double[K + 1];
            Y = new double[K + 1];
            Z = new double[K + 1];
            int id = 0;
            for (int i = 0; i < PTS.Length; i += 3)
            {
                X[id] = PTS[i];
                Y[id] = PTS[i + 1];
                Z[id] = PTS[i + 2];
                id++;
            }
            V0 = Parameters[12 + A + 4 * K];
            V1 = Parameters[13 + A + 4 * K];
            XNORM = Parameters[14 + A + 4 * K];
            YNORM = Parameters[15 + A + 4 * K];
            ZNORM = Parameters[16 + A + 4 * K];
            // using the Cox-de Boor formula to calculate the basis functions
            B = new BSpline_Basis_Function(T, M + 1);
		}

		int K; // Upper index of sum.
		int M; // Degree of basis functions
		int PROP1; // 0 = nonplanar, 1 = planar
		int PROP2; // 0 = open curve, 1 = closed curve
		int PROP3; // 0 = rational, 1 = polynomial
		int PROP4; // 0 = nonperiodic, 1 = periodic
		double[] T; // knot sequence
		double[] W; // Weights
		double[] X; // X control points
        double[] Y; // Y 
        double[] Z; // Z
        double V0; // starting parameter value
        double V1; // ending parameter value 
        // unit normal (if curve is planar)
        double XNORM;
        double YNORM;
        double ZNORM;
        BSpline_Basis_Function B;
        TransformationMatrix R;
        int ParameterId;

        public double3[] Evaluate(int Numpoints) {
            var Pts = new double3[Numpoints];
            var t = V0;
            var dt = (V1 - V0) / (Numpoints - 1);
            for (int iPt = 0; iPt < Numpoints; iPt++) {
                Pts[iPt] = EvalHelper(t);
                t += dt;
            }
            return Pts;
        }
        private double3 EvalHelper(double Val) {
            double hNsum = 0;
            double[] Nk = new double[W.Length];
            for (int i = 0; i < Nk.Length; i++)
            {
                Nk[i] = B.Polys[i].Evaluate(Val);
                hNsum += W[i] * Nk[i];
            }
            double[] Rk = new double[W.Length];
            double Checksum = 0;
            for (int i = 0; i < W.Length; i++)
            {
                Rk[i] = W[i] * Nk[i] / hNsum;
                Checksum += Rk[i];
            }
            var Ans = new double3(0);
            for (int i = 0; i < W.Length; i++)
            {
                Ans.x += X[i] * Rk[i];
                Ans.y += Y[i] * Rk[i];
                Ans.z += Z[i] * Rk[i];
            }
            return Ans;
        }

    };
}

