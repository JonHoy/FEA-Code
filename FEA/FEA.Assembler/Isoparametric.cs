using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace FEA.Assembler
{
    public class Isoparametric
    {
		public static double[,] GetCoefficientMatrix (int[] Order) {
			int NumTerms = 1;
			for (int i = 0; i < Order.Length; i++) {
				NumTerms = NumTerms * Order [i];
			}
			int Dims = Order.Length;
			var C = new double[NumTerms, NumTerms];
			var Pts = new double[NumTerms, Dims];
			var Idx = new Index (Order);
			for (int i = 0; i < NumTerms; i++) {
				var Sub = Idx.Ind2Sub (i);
				for (int j = 0; j < Dims; j++) {
					Pts [i, j] = (((double)Sub [j]) / (double)(Order[j] - 1)) * 2.0 - 1.0;
				}
			}
			for (int i = 0; i < NumTerms; i++) {
				for (int j = 0; j < NumTerms; j++) {
					var Sub = Idx.Ind2Sub (j);
					C [i, j] = 1;
					for (int k = 0; k < Dims; k++) {
						C [i, j] = C [i, j] * Math.Pow (Pts [i, k], (double)Sub [k]); 
					}
				}
			}
			// once this Coefficient matrix is found we must invert it for it to be useful
			Matrix<double> Cmat = DenseMatrix.OfArray (C);
			Cmat = Cmat.Inverse ();
			return Cmat.ToArray();
		}	

	}
}
