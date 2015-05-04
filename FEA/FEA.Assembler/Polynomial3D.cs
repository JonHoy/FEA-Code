using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FEA
{
    public class Polynomial3D
    {
        public Polynomial3D(double[,,] Coefficients) {
            coefficients = Coefficients;
            xOrder = Coefficients.GetLength(0) - 1;
            yOrder = Coefficients.GetLength(1) - 1;
            zOrder = Coefficients.GetLength(2) - 1;
        }
        public Polynomial3D(int Xorder, int Yorder, int Zorder)
        {
            yOrder = Yorder;
            xOrder = Xorder;
            zOrder = Zorder;
            coefficients = new double[xOrder + 1, yOrder + 1, zOrder + 1];
        }
        public int xOrder;
        public int yOrder;
        public int zOrder;
        public double[, ,] coefficients;    
        public static Polynomial3D operator +(Polynomial3D p2, Polynomial3D p1) {
            var coefficients = addArrays(p2.coefficients, p1.coefficients);
            var p3 = new Polynomial3D(coefficients);
            return p3;
        }
        public static Polynomial3D operator -(Polynomial3D p2, Polynomial3D p1) {
            return p2 + p1 * -1.0;
        }
        public static Polynomial3D operator *(Polynomial3D p2, Polynomial3D p1) {
            var p3 = new Polynomial3D(p2.xOrder + p1.xOrder, p2.yOrder + p1.yOrder, p2.zOrder + p1.zOrder);
            for (int i1 = 0; i1 <= p1.xOrder; i1++)
            {
                for (int i2 = 0; i2 <= p2.xOrder; i2++)
                {
                    for (int j1 = 0; j1 <= p1.yOrder; j1++)
                    {
                        for (int j2 = 0; j2 <= p2.yOrder; j2++)
                        {
                            for (int k1 = 0; k1 <= p1.zOrder; k1++)
                            {
                                for (int k2 = 0; k2 <= p2.zOrder; k2++)
                                {
                                    p3.coefficients[i1 + i2, j1 + j2, k1 + k2] += p2.coefficients[i2, j2, k2] * p1.coefficients[i1, j1, k1];
                                }
                            }                           
                        }
                    }
                }
            }
            return p3;
        }
		public static Polynomial3D operator *(Polynomial3D p1, double Scalar){
            var p2 = new Polynomial3D(p1.xOrder, p1.yOrder, p1.zOrder);
            for (int i = 0; i <= p2.xOrder; i++)
            {
                for (int j = 0; j <= p2.yOrder; j++)
                {
                    for (int k = 0; k <= p2.zOrder; k++)
                    {
                        p2.coefficients[i, j, k] = Scalar * p1.coefficients[i, j, k];
                    }
                    
                }
            }
            return p2;
        }
		public static Polynomial3D operator *(double Scalar, Polynomial3D p1) {
			return p1 * Scalar;
		}
        public double Evaluate(double x, double y, double z) {
            double val = 0;
            for (int i = 0; i <= xOrder; i++)
            {
                for (int j = 0; j <= yOrder; j++)
                {
                    for (int k = 0; k <= zOrder; k++)
                    {
                        val = val + coefficients[i, j, k] * Math.Pow(x, i) * Math.Pow(y, j) * Math.Pow(z, k);
                    }                
                }
            }
            return val;
        }
        public Polynomial3D Evaluate(double Val, int Dim) // turn a 2d polynomial into a regular one
        {
            Polynomial3D pAns;
            if (Dim == 0)
                pAns = new Polynomial3D(0, yOrder, zOrder);
            else if (Dim == 1)
                pAns = new Polynomial3D(xOrder, 0, zOrder);
            else if (Dim == 2)
                pAns = new Polynomial3D(xOrder, yOrder, 0);
            else
                throw new Exception("Dimension must be 0 or 1");
            for (int i = 0; i <= xOrder; i++)
            {
                for (int j = 0; j <= yOrder; j++)
                {
                    for (int k = 0; k <= zOrder; k++)
                    {
                        if (Dim == 0)
                            pAns.coefficients[0, j, k] += pAns.coefficients[i, j, k] * Math.Pow(Val, i);
                        else if (Dim == 1)
                            pAns.coefficients[i, 0, k] += pAns.coefficients[i, j, k] * Math.Pow(Val, j);
                        else // Dim == 2
                            pAns.coefficients[i, j, 0] += pAns.coefficients[i, j, k] * Math.Pow(Val, k);
                    }
                }
            }
            return pAns;
        }
        public double Integrate(Point A, Point B) {
            return Integrate(B.x, A.x, B.y, A.y, B.z, A.z);
        }
        public Polynomial3D Integrate (double b, double a, int dim) {
			var pInt = Integrate (dim);
			return pInt.Evaluate (b, dim) - pInt.Evaluate (a, dim);
		}
		public Polynomial3D Integrate(double Val, int dim) {
			var pInt = Integrate (dim);
			return pInt.Evaluate (Val, dim);
		}
		public double Integrate(double x2, double x1, double y2, double y1, double z2, double z1)
        {
			var Poly = Integrate (x2, x1, 0).Integrate (y2, y1, 1).Integrate (z2, z1, 2);
            return Poly.coefficients[0, 0, 0];
        }
        public double Integrate(double x, double y, double z)
        {
            var Poly = Integrate(x, 0).Integrate(y, 1).Integrate(z, 2);
            return Poly.coefficients[0, 0, 0];
        }
        public Polynomial3D Integrate(int Dim) {
            int xOrderNew;
            int yOrderNew;
            int zOrderNew;
            if (Dim == 0)
            {
                xOrderNew = Math.Max(xOrder + 1,1);
                yOrderNew = yOrder;
                zOrderNew = zOrder;
            }
            else if (Dim == 1)
            {
                yOrderNew = Math.Max(yOrder + 1, 1);
                xOrderNew = xOrder;
                zOrderNew = zOrder;
            }
            else if (Dim == 2)
            {
                zOrderNew = Math.Max(zOrder + 1, 1);
                xOrderNew = xOrder;
                yOrderNew = yOrder;
            }
            else
            {
                throw new Exception("Dimension must be 0 or 1 or 2");
            }
            var pInt = new Polynomial3D(xOrderNew, yOrderNew, zOrderNew);
            for (int i = 0; i <= xOrderNew; i++)
            {
                for (int j = 0; j <= yOrderNew; j++)
                {
                    for (int k = 0; k <= zOrderNew; k++)
                    {
                        if (Dim == 0 && i > 0)
                            pInt.coefficients[i, j, k] = coefficients[i - 1, j, k] / ((double)i);
                        else if (Dim == 1 && j > 0)
                            pInt.coefficients[i, j, k] = coefficients[i, j - 1, k] / ((double)j);
                        else if (Dim == 2 && k > 0)
                            pInt.coefficients[i, j, k] = coefficients[i, j, k - 1] / ((double)k);
                    } 
                }
            }
            return pInt;
        }
        public Polynomial3D Differentiate(int Dim)
        {
            int xOrderNew;
            int yOrderNew;
            int zOrderNew;
            if (Dim == 0)
            {
                xOrderNew = Math.Max(xOrder - 1,0);
                yOrderNew = yOrder;
                zOrderNew = zOrder;
            }
            else if (Dim == 1)
            {
                yOrderNew = Math.Max(yOrder - 1, 0);
                xOrderNew = xOrder;
                zOrderNew = zOrder;
            }
            else if (Dim == 2)
            {
                zOrderNew = Math.Max(zOrder - 1, 0);
                xOrderNew = xOrder;
                yOrderNew = yOrder;
            }
            else
            {
                throw new Exception("Dimension must be 0 or 1 or 2");
            }
            var pDiff = new Polynomial3D(xOrderNew, yOrderNew, zOrderNew);
            for (int i = 0; i <= xOrderNew; i++)
            {
                for (int j = 0; j <= yOrderNew; j++)
                {
                    for (int k = 0; k <= zOrderNew; k++)
			        {
			            if (Dim == 0)
                            pDiff.coefficients[i, j, k] = coefficients[i + 1, j, k] * (i + 1);
                        else if (Dim == 1)
                            pDiff.coefficients[i, j, k] = coefficients[i, j + 1, k] * (j + 1);
                        else // Dim == 2
                            pDiff.coefficients[i, j, k] = coefficients[i, j, k + 1] * (k + 1);
			        }

                }
            }
            return pDiff;
        }
        public static double[,,] addArrays(double[,,] small, double[,,] big) {
            int xLength = Math.Max(small.GetLength(0), big.GetLength(0));
            int yLength = Math.Max(small.GetLength(1), big.GetLength(1));
            int zLength = Math.Max(small.GetLength(2), big.GetLength(2));
            var ans = new double[xLength, yLength, zLength];
            for (int i = 0; i < small.GetLength(0); i++)
            {
                for (int j = 0; j < small.GetLength(1); j++)
                {
                    for (int k = 0; k < small.GetLength(2); k++)
                    {
                        ans[i, j, k] = ans[i, j, k] + small[i, j, k];
                    }
                    
                }
            }
            for (int i = 0; i < big.GetLength(0); i++)
            {
                for (int j = 0; j < big.GetLength(1); j++)
                {
                    for (int k = 0; k < big.GetLength(2); k++)
                    {
                        ans[i, j, k] = ans[i, j, k] + big[i, j, k];
                    }
                    
                }
            }
            return ans;
        }
    }
}
