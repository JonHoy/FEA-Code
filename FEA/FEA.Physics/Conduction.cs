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
		public static double[,] ComputeMassMatrix(PolyMatrix N, Point A, Point B) {
			if (N.Rows > 1) 
				throw new Exception ("N must be 1 row x n cols");
			N = N.Transpose () * N;
			var Me = N.Integrate (A, B);
			return Me;
		}
		public static double[,] ComputeCapacitanceMatrix(PolyMatrix N, Point A, Point B, double rho, double cp) {
			var CapMatrix = ComputeMassMatrix (N, A, B);
			var Cp = rho * cp;
			for (int i = 0; i < N.Rows; i++) {
				for (int j = 0; j < N.Cols; j++) {
					CapMatrix[i, j] = CapMatrix[i, j] * Cp;
				}
			}
			return CapMatrix;
		}
		public static PolyMatrix ComputeGradientMatrix (PolyMatrix N)
		{
			if (N.Rows > 1) 
				throw new Exception ("N must be 1 row x n cols");
			var Ndx = N.Differentiate (0);
			var Ndy = N.Differentiate (1);
			var Ndz = N.Differentiate (2);
			var Gradient = new PolyMatrix (3, N.Cols);
			for (int j = 0; j < N.Cols; j++) {
				Gradient.Data[0, j] = Ndx.Data[0,j];
				Gradient.Data[1, j] = Ndy.Data[0,j];
				Gradient.Data[2, j] = Ndz.Data[0,j];
			}
			return Gradient;
		}

	}
}
