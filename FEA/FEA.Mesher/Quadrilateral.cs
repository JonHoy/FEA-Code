using System;
using ManagedCuda.VectorTypes;
using System.Collections.Generic;

namespace FEA.Mesher
{
    public class Quadrilateral
    {
        public Quadrilateral(double3 PointA, double3 PointB, double3 PointC, double3 PointD)
        {
            A = PointA;
            B = PointB;
            C = PointC;
            D = PointD;
        }
        // TODO account for nans by turning quadrilateral into triangle
        public double3 Intersection(double3 O, double3 D) {
             // to break up the quadrilateral into two triangles
            // Since a quadrilateral with two equal points is a triangle,
            // we have to figure out which (if any) of the points are equal
            var AB = A - B;
            if (AB.Length == 0)
                return ReturnTriIntersect(1, O, D);
            var BC = B - C;
            if (BC.Length == 0)
                return ReturnTriIntersect(2, O, D);
            var CD = C - D;
            if (CD.Length == 0)
                return ReturnTriIntersect(3, O, D);
            var DA = D - A;
            if (DA.Length == 0)
                return ReturnTriIntersect(4, O, D);
            var AC = A - C;
            if (AC.Length == 0)
                return ReturnTriIntersect(1, O, D);
            var BD = B - D;
            if (BD.Length == 0)
                return ReturnTriIntersect(2, O, D);
            var Ans = ReturnTriIntersect(1, O, D);
            if (double.IsNaN(Ans.x))
                Ans = ReturnTriIntersect(3, O, D);
            return Ans;
        }

        private double3 ReturnTriIntersect(int Exclude, double3 O, double3 D) {
            Triangle Tri;
            if (Exclude == 1)
                Tri = new Triangle(B,C,D);
            else if (Exclude == 2)
                Tri = new Triangle(A,C,D);
            else if (Exclude == 3)
                Tri = new Triangle(A,B,D);
            else if (Exclude == 4)
                Tri = new Triangle(A,B,C);
            else
                throw new Exception("Internal Error");
            var t = Tri.Intersection(O, D);

            return O + t * D;
        }

        public List<Triangle> Split() {
            // split this quadrilateral up into two non intersecting triangles
            // there is two possible valid configurations for the split,
            // the configuration chosen is the one with the maximum area/ perimeter ratio
            var Tri1 = new Triangle(A,B,C);
            var Tri2 = new Triangle(A, D, C);

            var Ratio1 = Tri1.Area() / Tri1.Perimeter() + Tri2.Area() / Tri2.Perimeter();
              
            var Tri3 = new Triangle(A, B, D);
            var Tri4 = new Triangle(C, B, D);

            var Ratio2 = Tri3.Area() / Tri3.Perimeter() + Tri4.Area() / Tri4.Perimeter();

            var Tris = new List<Triangle>(2);

            if (Ratio1 >= Ratio2)
            {
                Tris.Add(Tri1);
                Tris.Add(Tri2);
            }
            else
            {
                Tris.Add(Tri3);
                Tris.Add(Tri4);
            }
            return Tris;
        }

        public double3 A;
        public double3 B;
        public double3 C;
        public double3 D;



    }
}

