﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FEA.Assembler
{
    public class PolyMatrix {
        public PolynomialND[,] Data;
        public int Rows;
        public int Cols;
		public PolyMatrix(PolynomialND[,] Data) {
			Rows = Data.GetLength (0);
			Cols = Data.GetLength (1);
			this.Data = Data;
        }
		public PolyMatrix(int Rows, int Cols) {
			this.Rows = Rows;
			this.Cols = Cols;
			Data = new PolynomialND[Rows, Cols];
		}
        public static PolyMatrix operator +(PolyMatrix A, PolyMatrix B) { 
            if (A.Rows != B.Rows)
                throw new Exception("A and B must be equal in size");
            if (A.Cols != B.Cols)
                throw new Exception("A and B must be equal in size");
            var C = new PolyMatrix(A.Rows, A.Cols);
            for (int i = 0; i < A.Rows; i++)
            {
                for (int j = 0; j < B.Cols; j++)
                {
                    C.Data[i, j] = A.Data[i, j] + B.Data[i, j];
                }
            }
            return C;
        }
        public static PolyMatrix operator -(PolyMatrix A, PolyMatrix B) { // returns A - B 
            return -1 * B + A; 
        }
        public static PolyMatrix operator *(PolyMatrix A, PolyMatrix B)  // returns A * B
        {
            if (B.Rows != A.Cols)
                throw new Exception("Number of Columns of A must be equal to the Number of Rows of B");
            var C = new PolyMatrix(A.Rows, B.Cols);
            for (int i = 0; i < A.Rows; i++)
            {
                for (int j = 0; j < B.Cols; j++)
                {
                    for (int k = 0; k < A.Cols; k++)
                    {
                        C.Data[i,j] = A.Data[i, k] * B.Data[k, j];
                    }
                }
            }
            return C;
        }
		public static PolyMatrix operator *(PolyMatrix A, double[,] B) {
			if (B.GetLength(0) != A.Cols)
				throw new Exception("Number of Columns of A must be equal to the Number of Rows of B");
			var C = new PolyMatrix(A.Rows, B.GetLength(1));
			for (int i = 0; i < A.Rows; i++)
			{
				for (int j = 0; j < B.GetLength(1); j++)
				{
					for (int k = 0; k < A.Cols; k++)
					{
						C.Data[i,j] = A.Data[i, k] * B[k, j];
					}
				}
			}
			return C;
		}
		public static PolyMatrix operator *(double[,] A, PolyMatrix B) {
			if (B.Rows != A.GetLength(1))
				throw new Exception("Number of Columns of A must be equal to the Number of Rows of B");
			var C = new PolyMatrix(A.GetLength(0), B.Cols);
			for (int i = 0; i < A.GetLength(0); i++)
			{
				for (int j = 0; j < B.Cols; j++)
				{
					for (int k = 0; k < A.GetLength(1); k++)
					{
						C.Data[i,j] = A[i, k] * B.Data[k, j];
					}
				}
			}
			return C;
		}
		public PolyMatrix Transpose() {
			var polyT = new PolyMatrix(Cols, Rows);
            for (int i = 0; i < polyT.Rows; i++)
            {
                for (int j = 0; j < polyT.Cols; j++)
                {
                    polyT.Data[i, j] = Data[j, i];
                }
            }
            return polyT;
        }
        public double[,] Integrate(Point A, Point B) {
            var Ans = new double[Rows, Cols];
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Cols; j++)
                {
                    Ans[i, j] = Data[i, j].Integrate(A, B);
                }
            }
            return Ans;
        }
		public PolyMatrix Integrate(int Dim) {
			var Integral = new PolyMatrix(Data.GetLength (0), Data.GetLength (1));
			for (int i = 0; i < Rows; i++) {
				for (int j = 0; j < Cols; j++) {
					Integral.Data[i, j] = Data[i, j].Integrate(Dim);
				}
			}
			return Integral;
		}
		public PolyMatrix Differentiate(int Dim) {
			var gradient = new PolyMatrix (Rows, Cols);
			for (int i = 0; i < Rows; i++) {
				for (int j = 0; j < Cols; j++) {
					gradient.Data[i, j] = Data[i, j].Differentiate(Dim);
				}
			}
			return gradient;
		}
        public static PolyMatrix operator *(double Scalar, PolyMatrix A) { 
            var B = new PolyMatrix(A.Rows, A.Cols);
            for (int i = 0; i < A.Rows; i++)
            {
                for (int j = 0; j < A.Cols; j++)
                {
                    B.Data[i, j] = A.Data[i, j] * Scalar; 
                }
            }
            return B;
        }
		public static PolyMatrix operator *(PolyMatrix A, double Scalar) {
			return Scalar * A;
		}
    }
}
