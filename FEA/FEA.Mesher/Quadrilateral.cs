using System;
using ManagedCuda.VectorTypes;

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
            if (Ans.x == double.NaN)
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
            var Ans = Tri.Intersection(O, D);
            return Ans;
        }

        public void Split(out Triangle Triangle1, out Triangle Triangle2) {
            // split this quadrilateral up into two triangles
        }

        public double3 A;
        public double3 B;
        public double3 C;
        public double3 D;



    }
}

