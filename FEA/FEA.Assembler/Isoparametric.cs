﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System.IO;


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

		public static void WriteToCppFiles(int N = 3) {
			var Mat2 = new int[]{1, 1};
			var Mat3 = new int[]{1, 1, 1};
			var File = new List<string>();
			File.Add ("//Auto Generated File: " + DateTime.Now.ToLongDateString ());
			File.Add ("template<typename T, int Size, int Dims>");
			File.Add ("struct CoefficientMatrix{");
			File.Add ("__device__ CoefficientMatrix() {}");
			File.Add ("T C[Size][Size];");
			File.Add ("__device__ void GetAlpha(T A[Size], const T U[Size]) {");
			File.Add ("\t#pragma unroll");
			File.Add ("\tfor (int i = 0; i < Size; i++) {");
			File.Add ("\t\tT Sum = 0;");
			File.Add ("\t\t#pragma unroll");
			File.Add ("\t\tfor (int j = 0; j < Size; j++) {");
			File.Add ("\t\t\tSum = Sum + C[i][j] * U[j];"); 
			File.Add ("\t\t}");
			File.Add ("\t\tA[i] = Sum;");
			File.Add ("\t}");
			File.Add ("}");
			File.Add ("};");
			AddToFile (File, Mat2, N);
			AddToFile (File, Mat3, N);
			System.IO.File.WriteAllLines ("CoefficientMatrix.cuh", File.ToArray ());
		}
			
		static private void AddToFile(List<string> File, int[] Mat, int N) {
			for (int i = 0; i < N; i++) {
				Increment (Mat);
				var C2 = GetCoefficientMatrix (Mat);
				var L2 = Product (Mat);
				File.Add ("template<typename T>");
				File.Add ("struct CoefficientMatrix<T," + L2.ToString() + "," + Mat.Length.ToString() + "> {");
				File.Add ("__device__ CoefficientMatrix() {");
				File.Add ("C = {");
				for (int j = 0; j < L2; j++) {
					string Line = "{";
					for (int k = 0; k < L2; k++) {
						Line = Line + C2[j, k].ToString("E12");
						if (k < L2 - 1)
							Line = Line + ",";
					}
					Line = Line + "}";
					if (j < L2 - 1)
						Line = Line + ",";
					File.Add (Line);
				}
				File.Add ("}");
				File.Add ("}");
				File.Add ("};");
			}
		}
		static private void Increment(int[] A) {
			for (int i = 0; i < A.Length; i++) {
				A[i]++;
			}
		}
		static private int Product(int[] A) {
			var ans = 1;
			for (int i = 0; i < A.Length; i++) {
				ans = ans * A [i];
			}
			return ans;
		}

	}
}