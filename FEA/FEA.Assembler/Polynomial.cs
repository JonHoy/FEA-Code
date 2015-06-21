using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FEA.Assembler
{
    public class Polynomial
    {
        public Polynomial(double[] Coefficients) {
            coefficients = Coefficients;
            order = Coefficients.Length - 1;
        }
        public Polynomial(int Order)
        {
            order = Order;
            coefficients = new double[order + 1];
        }
        protected int order;      
        protected double[] coefficients;     
        public static Polynomial operator +(Polynomial p2, Polynomial p1) {
            int Order = Math.Max(p2.order, p1.order);
            var p3 = new Polynomial(Order);
            if (p2.order > p1.order)
            {
                p3.coefficients = addArrays(p1.coefficients, p2.coefficients);
            }
            else
            {
                p3.coefficients = addArrays(p2.coefficients, p1.coefficients);
            }
            return p3;
        }
        public static Polynomial operator -(Polynomial p2, Polynomial p1) {
            return p2 + p1 * -1.0;
        }
        public static Polynomial operator *(Polynomial p2, Polynomial p1) {
            Polynomial p3 = new Polynomial(p2.order + p1.order);
            for (int i = 0; i <= p1.order; i++)
            {
                for (int j = 0; j <= p2.order; j++)
                {
                    p3.coefficients[i + j] += p2.coefficients[j] * p1.coefficients[i];
                }
            }
            return p3;
        }
        public static Polynomial operator *(Polynomial p1, double Scalar){
            var p2 = new Polynomial(p1.order);
            for (int i = 0; i <= p1.order; i++)
			{
                p2.coefficients[i] = p1.coefficients[i] * Scalar;
			}
            return p2;
        }
        public double Evaluate(double x) {
            double ans = 0;
            for (int i = 0; i <= order; i++)
            {
                ans = ans + coefficients[i] * Math.Pow(x, i);
            }
            return ans;
        }
        public double Integrate(double x2, double x1) {
            Polynomial p1 = Integrate();
            return p1.Evaluate(x2) - p1.Evaluate(x1);
        }
        public Polynomial Integrate() {
            var ans = new Polynomial(order + 1);
            for (int i = 1; i <= ans.order; i++)
            {
                ans.coefficients[i] = (1.0d / (double)(i + 1)) * coefficients[i - 1];
            }
            return ans;
        }
        public Polynomial Differentiate() {
            int newOrder = order - 1;
            Polynomial ans;
            if (newOrder >= 0)
                ans = new Polynomial(newOrder);
            else
                ans = new Polynomial(0);
            for (int i = 0; i <= ans.order; i++)
            {
                ans.coefficients[i] = coefficients[i + 1] * (double)(i + 1);
            }
            return ans;
        }
		public PolynomialND Convert_ND(int Dim, int Rank = 3) {
			var Order = new int[Rank];
			for (int i = 0; i < Rank; i++)
				Order [i] = 1;
			Order [Dim] = coefficients.Length;
			var Ans = new PolynomialND (Order, coefficients);
			return Ans;
		}
        public Polynomial2D Convert_2D(int Dim) {
            Polynomial2D p2d;
            if (Dim == 0)
                p2d = new Polynomial2D(order, 0);
            else if (Dim == 1)
                p2d = new Polynomial2D(0, order);
            else
                throw new Exception("Dim must be 0, 1, or 2");
            int idx = 0;
            for (int i = 0; i <= p2d.xOrder; i++)
            {
                for (int j = 0; j <= p2d.yOrder; j++)
                {
                    p2d.coefficients[i, j] = coefficients[idx];
                    idx++;
                }
            }
            return p2d;
        }
        public Polynomial3D Convert_3D(int Dim) {
            Polynomial3D p3d;
            if (Dim == 0)
                p3d = new Polynomial3D(order, 0, 0);
            else if (Dim == 1)
                p3d = new Polynomial3D(0, order, 0);
            else if (Dim == 2)
                p3d = new Polynomial3D(0, 0, order);
            else
                throw new Exception("Dim must be 0, 1, or 2");
            int idx = 0;
            for (int i = 0; i <= p3d.xOrder; i++)
            {
                for (int j = 0; j <= p3d.yOrder; j++)
                {
                    for (int k = 0; k <= p3d.zOrder; k++)
                    {
                        p3d.coefficients[i, j, k] = coefficients[idx];
                        idx++;
                    }
                }
            }
            return p3d;
        }
        private static double[] addArrays(double[] small, double[] big) {
            var ans = new double[big.Length];
            for (int i = 0; i < big.Length; i++)
		    {
                ans[i] = big[i];			 
		    }
            for (int i = 0; i < small.Length; i++)
            {
                ans[i] = ans[i] + small[i];
            }
            return ans;
        }
        public static Polynomial[] LagrangeInterpolation(double[] x) { // construct an array of lagrangian interpolation functions
            var Polys = new Polynomial[x.Length];
            for (int i = 0; i < Polys.Length; i++)
                Polys[i] = new Polynomial(0);
                 
            for (int k = 0; k < x.Length; k++)
            {
                Polys[k].coefficients[0] = 1;
                double xk = x[k];
                for (int i = 0; i < x.Length; i++)
                {   
                    if (i != k) {
                        double xi = x[i];
                        var mPoly = new Polynomial(new double[] {-xi , 1}); // construct new multiplier polynomial
                        mPoly = mPoly * (1.0/(xk - xi));
                        Polys[k] = Polys[k] * mPoly; // multiply to the existing polynomial
                    }
                }
            }
            return Polys;
        }
    }
}
