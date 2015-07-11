using System;
using ManagedCuda.VectorTypes;
using MathNet.Numerics.LinearAlgebra;

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

            UnitNormal = new double3(a, b, c);
            UnitNormal.Normalize();

        } // construct a Plane that satisfies the following equality: X = Val or Y = Val or Z = Val
        // for X Dim == 0, Y Dim == 1, Z Dim == 2
        public Plane(double _a, double _b, double _c, double _d)
        {
            a = _a;
            b = _b;
            c = _c;
            d = _d;

            UnitNormal = new double3(a, b, c);
            UnitNormal.Normalize(); 
        }
        double a;
        double b;
        double c;
        double d;

        public double3 UnitNormal { get;}

        public Location AboveOrBelow(double3 Point) {
            double Val =  (a * Point.x + b * Point.y + c * Point.z - d);

            double Eps = 1.0e-7; // we must account for floating point errors by introducing a "Tolerance factor"
            // This effectively means we have to treat a singular value as that value +/- tolerance factor
            // For this reason, double precision is used and then it can be clamped by casting to single precision
            if (Val > Eps)
            {
                return Location.Above;
            }
            else if (Val < -Eps)
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

        private Matrix<double> TransformationMatrix(double3 x_new) {

            var z_old = new double3(0, 0, 1);
            var z_new = UnitNormal;
            var y_old = new double3(0, 1, 0);
            var x_old = new double3(1, 0, 0);
            x_new.Normalize(); // make sure its normalized
            if (z_new.Dot(x_new) != 0)
                throw new Exception("Axes must be orthonormal");
            var y_new = z_new.Cross(x_new);

            var Q = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.Build.Dense(3, 3);
            Q[0, 0] = x_new.Dot(x_old);
            Q[0, 1] = x_new.Dot(y_old);
            Q[0, 2] = x_new.Dot(z_old);

            Q[1, 0] = y_new.Dot(x_old);
            Q[1, 1] = y_new.Dot(y_old);
            Q[1, 2] = y_new.Dot(z_old);

            Q[2, 0] = z_new.Dot(x_old);
            Q[2, 1] = z_new.Dot(y_old);
            Q[2, 2] = z_new.Dot(z_old);

            return Q;
        }

        public double3 Transform(double3 Pt, double3 x_new) {
        /*
            Transforms a point such that the input normal vector of 0i + 0j + k
            is now alligned with the unit normal vector of the plane. This is is useful
            when you want to describe co-planar points in X-Y coordinates instead of X-Y-Z
            - Methods used 
            http://www.continuummechanics.org/cm/coordxforms.html
            v' = Q * v, v_i' = a_ij * v_j, a_ij = cos(x_i',x_j) 
            Where Q = 3 x 3 matrix
            v' is the output matrix 
            cos(x_i',x_j) is the direction cosine between the two axes

            z_old = [0 0 1];
            z_new = Unit_Normal;
            y_old = [0 1 0];
            y_new -> 
            x_old = [1 0 0];
            x_new = ->
        */


            // cos q = dot(v,w) / (mag(v) * mag(w)) 
            var Q = TransformationMatrix(x_new);
            var v = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.Build.Dense(3, 1);

            v[0,0] = Pt.x;
            v[1,0] = Pt.y;
            v[2,0] = Pt.z;

            var vprime = Q * v;

            return new double3(vprime[0,0], vprime[1,0], vprime[2,0]);

        }

        public double3 UnTransform(double3 Pt, double3 x_new) {
            //  Reverse of Transform Method
            var Q = TransformationMatrix(x_new);
            var vprime = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.Build.Dense(3, 1);

            vprime[0,0] = Pt.x;
            vprime[1,0] = Pt.y;
            vprime[2,0] = Pt.z;

            var v = Q.Inverse() * vprime;

            var Ans = new double3();
            Ans.x = v[0, 0];
            Ans.y = v[1, 0];
            Ans.z = v[2, 0];

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

