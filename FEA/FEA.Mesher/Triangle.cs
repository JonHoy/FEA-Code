using System;
using ManagedCuda.VectorTypes;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
namespace FEA.Mesher
{
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
        // Möller–Trumbore intersection algorithm (between ray and triangle)
        public double Intersection(double3 O, double3 D) {
            var Ans = new double3(double.NaN);
            double EPSILON = 0.000001; 
            var e1 = B - A;
            var e2 = C - A;
            var P = D.Cross(e2);
            var det = e1.Dot(P);

            if(det > -EPSILON && det < EPSILON) 
                return double.NaN;

            var inv_det = 1.0 / det;

            var T = O - A;
            var u = T.Dot(P) * inv_det;

            if(u < 0 || u > 1) 
                return double.NaN;

            var Q = T.Cross(e1);
            var v = D.Dot(Q) * inv_det;
            if (v < 0 || u + v > 1)
                return double.NaN;

            var t = e2.Dot(Q) * inv_det;

            if(t > EPSILON) { //ray intersection
                Ans = O + D * t;
            }  
            if (double.IsNaN(Ans.x))
                return double.NaN;
            else
                return t;
                
        }

        public bool Intersection(Triangle Tri2) {
            // determine if there is any overlap between the two triangles
            return false;
        }

//        public Sphere Circumcircle() {
//            // computes the circumcircle of this triangle
//        }

        public Plane ComputePlane() {
            var Mat = Matrix<double>.Build.Dense(3, 3); // the plane equation
            Mat[0,0] = A.x;
            Mat[1,0] = A.y;
            Mat[1,0] = A.z;

            Mat[0,1] = B.x;
            Mat[1,1] = B.y;
            Mat[2,1] = B.z;

            Mat[0,2] = C.x;
            Mat[1,2] = C.y;
            Mat[2,2] = C.z;

            var MatInv = Mat.Inverse();

            double d = 1;

            var f = Matrix<double>.Build.Dense(3, 1, -d);

            var Ans = MatInv * f;
            var a = Ans[0,0];
            var b = Ans[1,0];
            var c = Ans[2,0];

            return new Plane(a, b, c, d);

        }

        public Location AboveOrBelow(Plane Slice) {
            // determines if the triangle lies within, above, or below the plane
            // if all the elements are above or below the plane the answer is simple
            // if a triangle has nodes that lie below and lie above, it is within the plane
            // that triangle must be split up
            // if all the triangles points lie on plane, the triangle is within the plane
            // if 1 or 2 of the points lie within the plane the triangle belongs in the region
            // where the non planar point(s) are
            int AboveCount = 0;
            int BelowCount = 0;
            var LocA = Slice.AboveOrBelow(A);
            var LocB = Slice.AboveOrBelow(B);
            var LocC = Slice.AboveOrBelow(C);
            Location[] Locs = new Location[]{ LocA, LocB, LocC };
            foreach (var Loc in Locs)
            {
                if (Loc == Location.Above)
                {
                    AboveCount++;
                }
                else if (Loc == Location.Below)
                {
                    BelowCount++;
                }
            }
            if (AboveCount > 0 && BelowCount == 0)
            {
                return Location.Above;
            }
            else if (BelowCount > 0 && AboveCount == 0)
            {
                return Location.Below;
            }
            else
            {
                return Location.On;
            }
        }

        public bool InPlane(Plane Slice) {
            var Loc1 = (int)Slice.AboveOrBelow(A);
            var Loc2 = (int)Slice.AboveOrBelow(B);
            var Loc3 = (int)Slice.AboveOrBelow(C);
            if (Loc1 == 0 && Loc2 == 0 && Loc3 == 0)
                return true;
            else
                return false;
        }

        public void Split(Plane Slice, out List<Triangle> Above, out List<Triangle> Below) {
        // splits this triangle up along the plane into triangles that are either above or below the plane
            // if all the points are in the plane, the output for above and below is simply the triangle
            // if one point is in the plane, one is above the plane, and one is below the triangle is simply split into two
            // the split occurs at the edge which is shared by the above point and below point 

            // step 1: Compute intersection points by representing the edges as rays
            var Tris1 = new List<Triangle>();
            var Tris2 = new List<Triangle>();

            Above = new List<Triangle>();
            Below = new List<Triangle>();

            var O1 = A;
            var D1 = B - A; // t is also normalized from 0 to 1 automatically
            var O2 = B;
            var D2 = C - B;
            var O3 = C;
            var D3 = A - C;

            var P1 = Slice.Intersection(O1, D1, 1); // p1
            var P2 = Slice.Intersection(O2, D2, 1);
            var P3 = Slice.Intersection(O3, D3, 1);
            // TODO check for bisection 

            int Bisects = -1;

            if ((P1 - P2).Length == 0)
                Bisects = 1;
            else if ((P2 - P3).Length == 0)
                Bisects = 2;
            else if ((P1 - P3).Length == 0)
                Bisects = 3;
            if (Bisects == -1)
            {
                Quadrilateral Quad;
                if (double.IsNaN(P1.x))
                {
                    Quad = new Quadrilateral(P2, P3, A, B);
                    Tris1.Add(new Triangle(P2, P3, C));
                }
                else if (double.IsNaN(P2.x))
                {
                    Quad = new Quadrilateral(P1, P3, C, B);
                    Tris1.Add(new Triangle(P1, P3, A));
                }
                else
                {
                    Quad = new Quadrilateral(P1, P2, C, A);
                    Tris1.Add(new Triangle(P2, P1, B));
                }
                Tris2 = Quad.Split();

            }
            else
            {
                if (Bisects == 1)
                {
                    Tris1.Add(new Triangle(A, B, P3));
                    Tris2.Add(new Triangle(C, B, P3));
                }
                else if (Bisects == 2)
                {
                    Tris1.Add(new Triangle(A, C, P1));
                    Tris2.Add(new Triangle(C, B, P1));
                }
                else // Bisects == 3
                {
                    Tris1.Add(new Triangle(A, C, P2));
                    Tris2.Add(new Triangle(A, B, P2));
                }
            }
            var Loc = Tris1[0].AboveOrBelow(Slice);
            if (Loc == Location.Above)
            {
                Above = Tris1;
                Below = Tris2;
            }
            else
            {
                Above = Tris2;
                Below = Tris1;
            }

        }

        public double Area(){
            // herons formula
            var a = (A - B).Length;
            var b = (B - C).Length;
            var c = (C - A).Length;
            var s = (a + b + c)/ 2;
            var Area = Math.Sqrt(s * (s - a) * (s - b) * (s - c));
            return Area;
        }

        public double Perimeter() {
            var L1 = A - B;
            var L2 = B - C;
            var L3 = C - A;
            var Perimeter = L1.Length + L2.Length + L2.Length;
            return Perimeter;
        }

    }

}

