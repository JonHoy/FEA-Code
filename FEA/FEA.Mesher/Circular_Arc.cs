using System;

namespace FEA.Mesher.IGES
{
    public class Circular_Arc
    {
        public Circular_Arc(double[] Parameters)
        {
        }
        double ZT; // Parallel ZT displacement of arc from X T , Y T plane
        double X1; // Arc center abscissa
        double Y1; // Arc center ordinate
        double X2; // Start point abscissa
        double Y2; // Start point ordinate
        double X3; // Terminate point abscissa
        double Y3; // Terminate point ordinate
    }
}

