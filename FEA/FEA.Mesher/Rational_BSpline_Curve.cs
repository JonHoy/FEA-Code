using System;
using ManagedCuda.VectorTypes;

namespace FEA.Mesher
{
	public class Rational_BSpline_Curve
	{
		public Rational_BSpline_Curve (double[] Parameters)
		{
			K = (int) Parameters [1];
			M = (int)Parameters [2];
		}
		int K; // Upper index of sum.
		int M; // Degree of basis functions
		int PROP1; // 0 = nonplanar, 1 = planar
		int PROP2; // 0 = open curve, 1 = closed curve
		int PROP3; // 0 = rational, 1 = polynomial
		int PROP4; // 0 = nonperiodic, 1 = periodic
		double[] T; // knot sequence
		double[] W; // Weights
		double3[] PTS; // control points
		double3 NORM; // unit normal (if curve is planar)
	};
}

