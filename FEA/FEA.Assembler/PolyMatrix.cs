using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FEA
{
    public class PolyMatrix {
        public Polynomial3D[,] Data;
        public int Rows;
        public int Cols;
        public PolyMatrix(int Rows, int Cols) {
            this.Cols = Cols;
            this.Rows = Rows;
            Data = new Polynomial3D[Rows, Cols];
            for (int i = 0; i < Data.GetLength(0); i++)
            {
                for (int j = 0; j < Data.GetLength(1); j++)
                {
                    Data[i,j] = new Polynomial3D(0, 0, 0);
                }  
            }
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
        //public Polynomial3D Determinant();
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
    }
}
