using System;

namespace FEA.Mesher.IGES
{
    // Reference IGES SPEC v6.0 1998-01-05 pdf pp 166
    public class Rational_BSpline_Surface
    {
        public Rational_BSpline_Surface(double[] Parameters)
        {
            K1 = (int)Parameters[1];
            K2 = (int)Parameters[2];
            M1 = (int)Parameters[3];
            M2 = (int)Parameters[4];
            PROP1 = (int)Parameters[5];
            PROP2 = (int)Parameters[6];
            PROP3 = (int)Parameters[7];
            PROP4 = (int)Parameters[8];
            PROP5 = (int)Parameters[9];
            int N1 = 1 + K1 - M1;
            int N2 = 1 + K2 - M2;
            int A = N1 + 2 * M1;
            int B = N2 + 2 * M2;
            int C = (1 + K1) * (1 + K2);
        }
        int K1; // Upper index of first sum. See Appendix B
        int K2; // Upper index of second sum. See Appendix B
        int M1; // Degree of first set of basis functions
        int M2; // Degree of second set of basis functions
        int PROP1; // 1 = Closed in first parametric variable direction
        int PROP2; // 1 = Closed in second parametric variable direction
        int PROP3; // 0 = Rational, 1 = Polynomial
        int PROP4; // 1 = Periodic in first parametric variable direction
        int PROP5; // 1 = Periodic in second parametric variable direction
        double[] S; // first knot sequence
        double[] T; // second knot sequence
        double[,] W; // weights
        double[,] PTS; // Control Points
        double U0; // Starting value for first parametric direction
        double U1; // Ending value for first parametric direction
        double V0; // Starting value for second parametric direction
        double V1; // Ending value for second parametric direction
    }
}

