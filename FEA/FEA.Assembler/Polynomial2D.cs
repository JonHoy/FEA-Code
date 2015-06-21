using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FEA.Assembler
{
    public class Polynomial2D
    {
        public Polynomial2D(double[,] Coefficients) {
            coefficients = Coefficients;
            xOrder = Coefficients.GetLength(0) - 1;
            yOrder = Coefficients.GetLength(1) - 1;
        }
        public Polynomial2D(int Xorder, int Yorder)
        {
            yOrder = Yorder;
            xOrder = Xorder;
            coefficients = new double[xOrder + 1, yOrder + 1];
        }
        public int xOrder { get; private set; } 
        public int yOrder { get; private set; } 
        public double[,] coefficients { get; private set; }     
        public static Polynomial2D operator +(Polynomial2D p2, Polynomial2D p1) {
            var coefficients = addArrays(p2.coefficients, p1.coefficients);
            var p3 = new Polynomial2D(coefficients);
            return p3;
        }
        public static Polynomial2D operator -(Polynomial2D p2, Polynomial2D p1) {
            return p2 + p1 * -1.0;
        }
        public static Polynomial2D operator *(Polynomial2D p2, Polynomial2D p1) {
            var p3 = new Polynomial2D(p2.xOrder + p1.xOrder, p2.yOrder + p1.yOrder);
            for (int i1 = 0; i1 <= p1.xOrder; i1++)
            {
                for (int i2 = 0; i2 <= p2.xOrder; i2++)
                {
                    for (int j1 = 0; j1 <= p1.yOrder; j1++)
                    {
                        for (int j2 = 0; j2 <= p2.yOrder; j2++)
                        {
                            p3.coefficients[i1 + i2, j1 + j2] += p2.coefficients[i2, j2] * p1.coefficients[i1, j1];
                        }
                    }
                }
            }
            return p3;
        }
        public static Polynomial2D operator *(Polynomial2D p1, double Scalar){
            var p2 = new Polynomial2D(p1.xOrder, p1.yOrder);
            for (int i = 0; i <= p2.xOrder; i++)
            {
                for (int j = 0; j <= p2.yOrder; j++)
                {
                    p2.coefficients[i, j] = Scalar * p1.coefficients[i,j];
                }
            }
            return p2;
        }
        public double Evaluate(double x, double y) {
            double val = 0;
            for (int i = 0; i <= xOrder; i++)
            {
                for (int j = 0; j <= yOrder; j++)
                {
                    val = val + coefficients[i, j] * Math.Pow(x, i) * Math.Pow(y, j);
                }
            }
            return val;
        }
        public Polynomial2D Evaluate(double Val, int Dim) // turn a 2d polynomial into a regular one
        {
            Polynomial2D pAns;
            if (Dim == 0)
                pAns = new Polynomial2D(0, yOrder);
            else if (Dim == 1)
                pAns = new Polynomial2D(xOrder, 0);
            else
                throw new Exception("Dimension must be 0 or 1");
            
            for (int i = 0; i <= xOrder; i++)
            {
                for (int j = 0; j <= yOrder; j++)
                {
                    if (Dim == 0)
                        pAns.coefficients[0, j] += pAns.coefficients[i, j] * Math.Pow(Val, i);
                    else
                        pAns.coefficients[i, 0] += pAns.coefficients[i, j] * Math.Pow(Val, j);
                }
            }
            return pAns;
        }
        public double Integrate(double x2, double x1, double y2, double y1)
        {
            var pInt = Integrate(0);
            var pEval = pInt.Evaluate(x2, 0) - pInt.Evaluate(x1, 0); // reduces to a 1 x n array
            var pIInt = pEval.Integrate(1);
            var pEval2 = pIInt.Evaluate(y2, 1) - pIInt.Evaluate(y1, 1); // reduces to a 1 x 1 array
            return pEval2.coefficients[0, 0];
        }
        public Polynomial2D Integrate(int Dim) {
            int xOrderNew;
            int yOrderNew;
            if (Dim == 0)
            {
                xOrderNew = Math.Max(xOrder + 1,1);
                yOrderNew = yOrder;
            }
            else if (Dim == 1)
            {
                yOrderNew = Math.Max(yOrder + 1, 1);
                xOrderNew = xOrder;                
            }
            else
            {
                throw new Exception("Dimension must be 0 or 1");
            }
            var pInt = new Polynomial2D(xOrderNew, yOrderNew);
            for (int i = 0; i <= xOrderNew; i++)
            {
                for (int j = 0; j <= yOrderNew; j++)
                {
                    if (Dim == 0 && i > 0)
                        pInt.coefficients[i, j] = coefficients[i - 1, j] / ((double) i);  
                    else if(Dim == 1 && j > 0)
                        pInt.coefficients[i, j] = coefficients[i, j - 1] / ((double) j);
                }
            }
            return pInt;
        }
        public Polynomial2D Differentiate(int Dim)
        {
            int xOrderNew;
            int yOrderNew;
            if (Dim == 0)
            {
                xOrderNew = Math.Max(xOrder - 1,0);
                yOrderNew = yOrder;
            }
            else if (Dim == 1)
            {
                yOrderNew = Math.Max(yOrder - 1, 0);
                xOrderNew = xOrder;                
            }
            else
            {
                throw new Exception("Dimension must be 0 or 1");
            }
            var pDiff = new Polynomial2D(xOrderNew, yOrderNew);
            for (int i = 0; i <= xOrderNew; i++)
            {
                for (int j = 0; j <= yOrderNew; j++)
                {
                    if (Dim == 0)
                        pDiff.coefficients[i, j] = coefficients[i + 1, j] * (i + 1);
                    else
                        pDiff.coefficients[i, j] = coefficients[i, j + 1] * (j + 1);
                }
            }
            return pDiff;
        }
        public static double[,] addArrays(double[,] small, double[,] big) {
            int xLength = Math.Max(small.GetLength(0), big.GetLength(0));
            int yLength = Math.Max(small.GetLength(1), big.GetLength(1));
            var ans = new double[xLength, yLength];
            for (int i = 0; i < small.GetLength(0); i++)
            {
                for (int j = 0; j < small.GetLength(1); j++)
                {
                    ans[i, j] = ans[i, j] + small[i, j];
                }
            }
            for (int i = 0; i < big.GetLength(0); i++)
            {
                for (int j = 0; j < big.GetLength(1); j++)
                {
                    ans[i, j] = ans[i, j] + big[i, j];
                }
            }
            return ans;
        }
    }
}
