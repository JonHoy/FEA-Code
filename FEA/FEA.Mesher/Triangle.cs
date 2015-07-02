using System;
using ManagedCuda.VectorTypes;

namespace FEA.Mesher
{
    //[StructLayout(LayoutKind.Sequential)]
    public class Triangle
    {
        public Triangle(double3 PointA, double3 PointB, double3 PointC)
        {
            A = PointA;
            B = PointB;
            C = PointC;
        }

        public Triangle(double2 PointA, double2 PointB, double2 PointC)
        {
            A = new double3(PointA.x, PointA.y, 0);
            B = new double3(PointB.x, PointB.y, 0);
            C = new double3(PointC.x, PointC.y, 0);
        }


        public double3 A;
        public double3 B;
        public double3 C;
        // Möller–Trumbore intersection algorithm
        public double3 Intersection(double3 O, double3 D) {
            var Ans = new double3(double.NaN);
            double EPSILON = 0.000001; 
            var e1 = B - A;
            var e2 = C - A;
            var P = D.Cross(e2);
            var det = e1.Dot(P);

            if(det > -EPSILON && det < EPSILON) 
                return Ans;

            var inv_det = 1.0 / det;

            var T = O - A;
            var u = T.Dot(P) * inv_det;

            if(u < 0 || u > 1) 
                return Ans;

            var Q = T.Cross(e1);
            var v = D.Dot(Q) * inv_det;
            if (v < 0 || u + v > 1)
                return Ans;

            var t = e2.Dot(Q) * inv_det;

            if(t > EPSILON) { //ray intersection
                Ans = O + D * t;
            }  
            return Ans;
        }

    }
}

