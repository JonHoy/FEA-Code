using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FEA;
using System.Diagnostics;

namespace Unit_Tests
{
    class Program
    {
        static void Main(string[] args)
        {
            // Polynomial Definitions
            #region Polynomial
            // multiplication test
            var p1 = new Polynomial(new double[] { 1, -1 });
            var p2 = p1 * p1;
            var p3 = p2 * p2; 
            Debug.Assert(p3.Evaluate(1) == 0, "A Polynomial must be zero at one of its roots");
            var pminus = p1 - p1;
            Debug.Assert(pminus.Evaluate(1) == 0, "A Polynomial minus itself must be zero");
            var X = new double[] { 1, 2, 3, 4 };
            var LPolys = Polynomial.LagrangeInterpolation(X);
            for (int j = 0; j < LPolys.Length; j++)
            {
                for (int i = 0; i < LPolys.Length; i++)
                {
                    double Val = LPolys[i].Evaluate(X[j]);
                    Val = Math.Round(Val, 4); // Account for Round off error
                    if (i == j)
                        Debug.Assert(Val == 1, "Local Shape Function Must be equal to 1 at its own Node!");
                    if (i != j)
                        Debug.Assert(Val == 0, "Local Shape Function Must be equal to 0 at other Nodes!");
                }
            }     
            #endregion
            // Shape Function Definition Test
            #region ShapeFunction 
            var B = new Point();
            int Order = 2;
            B.x = 1; B.y = 1; B.z = 1; // unit cube
            var N = ShapeFunction.Generate(B, Order);
            // Definition 1: Ni(xi,yi,zi) = 1
			var NodeCount = Math.Pow((double)(1 + Order),3.0);
			var NumPts = (int) NodeCount;
			var XPts = new double[NumPts];
            var YPts = new double[NumPts];
            var ZPts = new double[NumPts];
            int idx = 0;
            for (int i = 0; i <= Order; i++)
            {
                for (int j = 0; j <= Order; j++)
                {
                    for (int k = 0; k <= Order; k++)
                    {
                        XPts[idx] = B.x * ((double)i / (double)Order);
                        YPts[idx] = B.y * ((double)j / (double)Order);
                        ZPts[idx] = B.z * ((double)k / (double)Order);
                        double Value = N.Data[0, idx].Evaluate(XPts[idx], YPts[idx], ZPts[idx]);
                        Value = Math.Round(Value, 4); // Account for Round off Error
                        Debug.Assert(Value == 1, "By Definition this must be one!");
                        idx++;
                    }
                }
            }          
            // Definition 3: Ni(x,y,z) = 0 if (x,y,z) = Other Node Points 
            #endregion
            #region CUDA LINEAR Solver
            Console.WriteLine("Cuda LinearSolver");
            int[] I;
            int[] J;
            float[] val;
            float[] res;
            int NumRows = (int)10e6;
            int nz = (NumRows-2)*3 + 4;
            genTridiag(out I, out J, out val, out res, NumRows, nz);
            var Solution = LinearSolver.GPU_Single_ConjugateGradient(J, I, val, res, new float[NumRows]);
            #endregion
        }
        static private void genTridiag(out int[] I, out int[] J, out float[] val, out float[] res, int N, int nz) {
            I = new int[N + 1];
            J = new int[nz];
            val = new float[nz];
            res = new float[N];
            for (int i = 0; i < N; i++)
            {
                res[i] = 1;
            }
            var Rng = new Random();
            float RAND_MAX = 32767;
            I[0] = 0; J[0] = 0; J[1] = 1;
            val[0] = (float)Rng.NextDouble()/RAND_MAX + 10.0f;
            val[1] = (float)Rng.NextDouble()/RAND_MAX;
            int start;

            for (int i = 1; i < N; i++)
            {
                if (i > 1)
                {
                    I[i] = I[i-1]+3;
                }
                else
                {
                    I[1] = 2;
                }

                start = (i-1)*3 + 2;
                J[start] = i - 1;
                J[start+1] = i;

                if (i < N-1)
                {
                    J[start+2] = i + 1;
                }

                val[start] = val[start-1];
                val[start+1] = (float)Rng.NextDouble()/RAND_MAX + 10.0f;

                if (i < N-1)
                {
                    val[start+2] = (float)Rng.NextDouble()/RAND_MAX;
                }
            }

            I[N] = nz;
        } 
    }
}
