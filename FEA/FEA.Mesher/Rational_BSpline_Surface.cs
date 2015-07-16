using System;
using ManagedCuda.VectorTypes;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace FEA.Mesher.IGES
{
    // Reference IGES SPEC v6.0 1998-01-05 pdf pp 166

    //In the Far Future: (Could NURBS be used to computationally describe waves of turbulence?) ie Isosurface as a function of time
    //In the Near Future: (Could NURBS be used to Section off sub domains of the part to utilize parallelism?)
    // Break a part into subparts and mesh them on the gpu
    // Use the 3D Point in Polygon on Wireframed NURBS?
    // Use coarse enough level of wireframe to fit in gpu cache and iteratively refine the region until some tolerance is reached
    // Create a NURB object using C++ templates and static arrays -> using meta programming and nvcc instatiate the template at runtime
    // Why NURBS are attractive for CFD/ Turbulence Research and GPU Computing:
    // -> Very Accurate Surface modeling (Boundary Layer)
    // -> Not constrained to triangular meshing
    // -> Can map nicely to hexahderal isoparametric elements (This allows for adaptive mesh refinement procedures
    // -> Low Memory Footprint (Can exploit caching)
    // -> High Amount of FLOPS / Byte (Able to exploit GPU efficiency) 
    // Drawbacks
    // -> Computing surface intersection of a ray can be computationally expensive (RESEARCH THIS)
    // -> Unlike a normal flat plane there might be multiple intersection points for the ray on the surface
    // -> Edges might not join together like they should (Surface Gaps in the CAD model, The Planes might intersect each other outside the edges)
    // -> Since C
    public class Rational_BSpline_Surface
    {
        public Rational_BSpline_Surface(double[] Parameters, TransformationMatrix _R = null, int _ParameterId = -1)
        {
            R = _R;
            ParameterId = _ParameterId;
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
            Bi = new BSpline_Basis_Function(S, M1 + 1);
            Bj = new BSpline_Basis_Function(T, M2 + 1);
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
        TransformationMatrix R; // Matrix that rotates and translates the surface to the correct position
        int ParameterId;

        public double3[,] Evaluate(int NumpointsU, int NumpointsV) {
            var Pts = new double3[NumpointsU, NumpointsV];
            var u = U0;
            var du = (U1 - U0) / (NumpointsU - 1);
            var dv = (V1 - V0) / (NumpointsV - 1);
            for (int i = 0; i < NumpointsU; i++) {
                var v = V0;
                for (int j = 0; j < NumpointsV; j++)
                {
                    Pts[i, j] = EvalHelper(u, v);
                    v += dv;
                }
                u += du;
            }
            return Pts;
        }

        private double3 EvalHelper(double u, double v) { // Finds the Surface at value at the specified u and v values
            double hNMsum = 0;
            double3 Ans = new double3(0);
            int id = 0;
            var Nk = new double[K1 + 1];
            var Ml = new double[K2 + 1];
            for (int i = 0; i < Nk.Length; i++)
            {
                Nk[i] = Bi.Polys[i].Evaluate(u);
            }
            for (int i = 0; i < Ml.Length; i++)
            {
                Ml[i] = Bj.Polys[i].Evaluate(v);
            }
            for (int i = 0; i < Nk.Length; i++)
            {
                for (int j = 0; j < Ml.Length; j++) {
                    hNMsum += W[id] * Nk[i] * Ml[j];
                    id++;
                }
            }
            id = 0;
            for (int i = 0; i < Nk.Length; i++)
            {
                for (int j = 0; j < Ml.Length; j++) {
                    double Sij = W[id] * Nk[i] * Ml[j] / hNMsum; 
                    Ans.x += X[id] * Sij;
                    Ans.y += Y[id] * Sij;
                    Ans.z += Z[id] * Sij;
                    id++;
                }
            }
            return Ans;
        }


        private double3 IntersectionHelper(double3 d, double3 P, double2 InitialGuess) {
            // reference
            if (double.IsNaN(InitialGuess.x)) //( X field = u value, Y field = v value 
                return new double3(double.NaN);
            var nhat0 = new double3();
            if ((Math.Abs(d.x) > Math.Abs(d.y)) && (Math.Abs(d.x) > Math.Abs(d.z)))
            {
                nhat0.x = d.y;
                nhat0.y = -1.0 * d.x;
                nhat0.z = 0;
            }
            else
            {
                nhat0.x = 0;
                nhat0.y = d.z;
                nhat0.z = -1.0 * d.y;
            }
            var nhat1 = nhat0.Cross(d);
            double d0 = -1.0 * nhat0.Dot(P);
            double d1 = -1.0 * nhat1.Dot(P);

            // find the partial derivatives of S with respect to u and v

            double du = (U1 - U0) * 0.00001;
            double dv = (U1 - U0) * 0.00001;
            var Jacobian = Matrix<double>.Build.Dense(2, 2);           
            double Tolerance = 1.0e-6;
            int MaxIterations = 10;
            var Fuv = Matrix<double>.Build.Dense(2, 1);
            var uv = Matrix<double>.Build.Dense(2, 1);
            uv[0,0] = InitialGuess.x;
            uv[1,0] = InitialGuess.y;
            int iterationCount = 0;
            double ErrorEstimate = 1;
            while (iterationCount < MaxIterations && Tolerance < ErrorEstimate)
            {
                double u = uv[0,0];
                double v = uv[1,0];
                double3 Su = (EvalHelper(u + du, v) - EvalHelper(u - du, v)) / (2 * du);     
                double3 Sv = (EvalHelper(u, v + dv) - EvalHelper(u, v + dv)) / (2 * dv);
                Jacobian[0, 0] = nhat0.Dot(Su);
                Jacobian[0, 1] = nhat0.Dot(Sv);
                Jacobian[1, 0] = nhat1.Dot(Su);
                Jacobian[1, 1] = nhat1.Dot(Sv);
                double3 S = EvalHelper(u, v);
                Fuv[0,0] = nhat0.Dot(S) + d0;
                Fuv[1,0] = nhat1.Dot(S) + d1;
                var uvnew = uv - Jacobian.Inverse() * Fuv;
                ErrorEstimate = (uvnew - uv).L1Norm();
                    
                iterationCount++;
                uv = uvnew;
            }
            if (iterationCount >= MaxIterations || Tolerance < ErrorEstimate)
                return new double3(double.NaN);
            else
                return EvalHelper(uv[0,0], uv[1,0]);
        }
          



        // Ray = P + td (Where P is the origin and d is the direction of the ray)
        public double3[] Intersection(double3 d, double3 P) {
            var ControlNet = Evaluate(2* (K1 + 1), 2 * (K2 + 1));
            Quadrilateral Quad;
            var IntesectionPts = new List<double3>();
            int Rows = ControlNet.GetLength(0) - 1;
            int Cols = ControlNet.GetLength(1) - 1;
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Cols; j++) {
                    Quad = new Quadrilateral(ControlNet[i, j], ControlNet[i, j + 1], ControlNet[i + 1, j + 1], ControlNet[i + 1, j]);
                    var LocalIntersection = Quad.Intersection(P, d);
                    if (!double.IsNaN(LocalIntersection.x))
                    {
                        var InitialGuess = new double2();
                        InitialGuess.x = (U1 - U0) * ((i + 0.5 )/ (double) Rows) + U0;
                        InitialGuess.y = (V1 - V0) * ((j + 0.5) / (double) Cols) + V0;
                        var Point = IntersectionHelper(d, P, InitialGuess);
                        if (!double.IsNaN(Point.x))
                            IntesectionPts.Add(Point);
                    }
                }
            }
            return IntesectionPts.ToArray();
        }
    }
}

