using System;
using System.Linq;
using System.Collections.Generic;

namespace FEA.Assembler
{
    public class Piecewise // this class implements piecewise polynomials over the domain [-inf, inf]
    {
        // The value of the piecewise polynomial is zero between [-inf , start] and [end, inf]
        public Piecewise(double Start, double End, Polynomial P)
        {
            if (!(End > Start))
                throw new Exception("End must be greater than Start");
            x = new double[]{ Start, End };
            Polynomials = new Polynomial[1];
            Polynomials[0] = P;
        }
        public Piecewise(double[] Points, Polynomial[] Polys) {
            if ((Points.Length - Polys.Length) != 1)
                throw new Exception("Points must be 1 element more than Polys");
            Polynomials = Polys;
            x = Points; 
        }
        public static Piecewise operator+(Piecewise p1, Piecewise p2) {
            var x = MakeVector(p1, p2);
            var Polys = new Polynomial[x.Length - 1];
            for (int i = 0; i < Polys.Length; i++)
            {
                double Val = (x[i] + x[i + 1]) / 2.0;
                int id1 = p1.Index(Val);
                int id2 = p2.Index(Val);
                if (id1 > -1 && id2 > -1)
                {
                    Polys[i] = p1.Polynomials[id1] + p2.Polynomials[id2];
                }
                else if (id1 > -1) {
                    Polys[i] = p1.Polynomials[id1];
                }
                else if (id2 > -1) {
                    Polys[i] = p2.Polynomials[id2];
                }
                else
                {
                    Polys[i] = new Polynomial(new double[]{ 0 });
                }
            }
            return new Piecewise(x, Polys);
        }
        public static Piecewise operator-(Piecewise p1, Piecewise p2) {
            p2 = p2 * -1.0;
            return p1 + p2;
        }
        public static Piecewise operator*(Piecewise p1, Piecewise p2) {
            var xnew = MakeVector(p1, p2);
            var Polys = new Polynomial[xnew.Length - 1];
            for (int i = 0; i < Polys.Length; i++)
            {
                double Val = (xnew[i] + xnew[i + 1]) / 2.0;
                int id1 = p1.Index(Val);
                int id2 = p2.Index(Val);
                if (id1 > -1 && id2 > -1)
                {
                    Polys[i] = p1.Polynomials[id1] * p2.Polynomials[id2];
                }
                else
                {
                    Polys[i] = new Polynomial(new double[]{ 0 });
                }
            }
            return new Piecewise(xnew, Polys);
        }
        public static Piecewise operator*(Piecewise p1, Polynomial p) {
            var pnew = new Polynomial[p1.x.Length - 1];
            var xnew = p1.x;
            for (int i = 0; i < pnew.Length; i++)
            {
                pnew[i] = p1.Polynomials[i] * p;
            }
            return new Piecewise(xnew, pnew);
        }
        public static Piecewise operator*(Piecewise p1, double a) {
            var pnew = new Polynomial[p1.x.Length - 1];
            var xnew = p1.x;
            for (int i = 0; i < pnew.Length; i++)
            {
                pnew[i] = p1.Polynomials[i] * a;
            }
            return new Piecewise(xnew, pnew);
        }

        private static double[] MakeVector(Piecewise p1, Piecewise p2) {
            var xnew = new HashSet<double>();
            for (int i1 = 0; i1 < p1.x.Length; i1++)
            {
                xnew.Add(p1.x[i1]);
            }
            for (int i2 = 0; i2 < p2.x.Length; i2++)
            {
                xnew.Add(p2.x[i2]);
            }
            var x = xnew.ToArray();
            Array.Sort(x);
            return x;
        }
        public int Index(double Val) {
            // if in bounds it returns the index number of the specified value else
            int Id = -1;
            for (int i = 1; i < x.Length; i++)
            {
                if (Val <= x[i] && Val >= x[i - 1])
                {
                    Id = i - 1;
                    break;
                }
            }
            return Id;
        }
        public double Evaluate(double Val) { // evaluates the piecewise polynomial
            // discontinuities evaluate on the negative side 
            // ie (Y = a -> [-1 , 1] and Y = b [1 , 2]) thus, Y(1) = a
            double Ans = 0;
            int Location = Index(Val);
            if (Location != -1)
            {
                Ans = Polynomials[Location].Evaluate(Val);
            }
            return Ans;
        }
        public void Simplify() {
            // this function removes Polynomials on the edges of the domain which equal zero
        }
        double[] x; // monotomically unique increasing vector
        Polynomial[] Polynomials;
    }
}

