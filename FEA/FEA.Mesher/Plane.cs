using System;
using ManagedCuda.VectorTypes;
namespace FEA.Mesher
{
    public class Plane // equation of a plane (ax + by + cz = d)
    {
        public Plane(double Val, int Dim) {
            d = Val;
            a = 0;
            b = 0;
            c = 0;
            if (Dim == 0)
            {
                a = 1;
            }
            else if (Dim == 1)
            {
                b = 1;
            }
            else if (Dim == 2)
            {
                c = 1;
            }
            else 
            {
                throw new Exception("Dim must be 0 , 1, or 2"); 
            }
        } // construct a Plane that satisfies the following equality: X = Val or Y = Val or Z = Val
        // for X Dim == 0, Y Dim == 1, Z Dim == 2
        public Plane(double _a, double _b, double _c, double _d)
        {
            a = _a;
            b = _b;
            c = _c;
            d = _d;
        }
        double a;
        double b;
        double c;
        double d;

        public Location AboveOrBelow(double3 Point) {
            double Val = a * Point.x + b * Point.y + c * Point.z - d;
            if (Val > 0)
            {
                return Location.Above;
            }
            else if (Val < 0)
            {
                return Location.Below;
            }
            else
            {
                return Location.On;
            }
        }

        public double3 Intersection(double3 O, double3 D, double tmax = double.PositiveInfinity) {
            // calculates the intersection of a ray and a plane
            // if no intersection occurs a vector of NaNs is returned

            var PlaneVec = new double3();
            PlaneVec.x = a;
            PlaneVec.y = b;
            PlaneVec.z = c;

            double t = (d - PlaneVec.Dot(O)) / PlaneVec.Dot(D);

            var Ans = new double3(double.NaN);
            if (t >= 0 && t <= tmax)
            {
                Ans = O + D * t;
            }
            return Ans;
        } 

    }
    // a point in space can either be on, below, or above the plane
    // if ax + by + cz = d the point is on the plane
    // if ax + by + cz < d the point is below the plane
    // if ax + by + cz > d the point is above the plane
    public enum Location {
        Below = -1,
        On = 0,
        Above = 1
    }


}

