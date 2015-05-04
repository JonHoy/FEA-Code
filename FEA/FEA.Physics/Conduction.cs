using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FEA;

namespace FEA.Physics
{
    class Conduction
    {
		public static PolyMatrix ComputeGradientMatrix (PolyMatrix N)
		{
			if (N.Rows > 1) 
				throw new Exception ("N must be 1 row x n cols");
			var Ndx = N.Differentiate (0);
			var Ndy = N.Differentiate (1);
			var Ndz = N.Differentiate (2);
			var B = new PolyMatrix (3, N.Cols);
			for (int j = 0; j < N.Cols; j++) {
				B.Data[0, j] = Ndx.Data[0,j];
				B.Data[1, j] = Ndy.Data[0,j];
				B.Data[2, j] = Ndz.Data[0,j];
			}
			return B;
		}
        // For Volume Elements
        #region Volume Terms 
		public static double[,] ComputeMassMatrix(PolyMatrix N, Point a, Point b) {
			if (N.Rows > 1) 
				throw new Exception ("N must be 1 row x n cols");
			N = N.Transpose () * N;
			var Me = N.Integrate (a, b);
			return Me;
		}
		public static double[,] ComputeCapacitanceMatrix(double[,] MassMatrix, double rho, double cp) {
			double Cp = rho * cp;
			int Rows = MassMatrix.GetLength(0);
			int Cols = MassMatrix.GetLength(1);
			var CapacitanceMatrix = new double[Rows, Cols];
			for (int i = 0; i < Rows; i++) {
				for (int j = 0; j < Cols; j++) {
					CapacitanceMatrix [i, j] = MassMatrix [i, j] * Cp;
				}
			}
			return CapacitanceMatrix;
		}
		public static double[,] ComputeCapacitanceMatrix(PolyMatrix N, Point a, Point b, double rho, double cp) {
			var CapMatrix = ComputeMassMatrix (N, a, b);
			var Cp = rho * cp;
			for (int i = 0; i < N.Rows; i++) {
				for (int j = 0; j < N.Cols; j++) {
					CapMatrix[i, j] = CapMatrix[i, j] * Cp;
				}
			}
			return CapMatrix;
		}
		public static double[,] ComputeConductionMatrix(PolyMatrix B, Point a, Point b, double k) {
			return ComputeConductionMatrix (B, a, b, k, k, k);
		}
		public static double[,] ComputeConductionMatrix(PolyMatrix B, Point a, Point b, double kx, double ky, double kz) {
			double[,] D = new double[3, 3]; 
			D [0, 0] = kx;
			D [1, 1] = ky;
			D [2, 2] = kz;
			var Kn = B.Transpose() * D * B;
			var Ke = Kn.Integrate(a, b);
			return Ke;
		}
		public static double[] ComputeHeatGenerationMatrix(double G, PolyMatrix N, Point a, Point b) {
			var P = N.Transpose ().Integrate (a, b);
			var P1 = new double[P.Length];
			for (int i = 0; i < P1.Length; i++) {
				P1 [i] = G * P [i, 0];
			}
			return P1;
		}
		#endregion 
        // For Surface Elements
        #region Surface Terms 
		//public static double[] ComputeConvectionMatrix
		//public static double[] ComputeHeatFluxMatrix
		//public static double[] ComputeRadiantConvectionMatrix
		#endregion
	}

}
