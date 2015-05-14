using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FEA
{
	public class PolynomialND
	{
		public PolynomialND(int[] Order) {
			this.Rank = Order.Length;
			this.Order = Order;
			int Count = 1;
			for (int i = 0; i < Rank; i++) {
				Count = Count * this.Order[i];
			}
			Coefficients = new double[Count];
			PolynomialBase (this.Order, Coefficients);
		}
		public PolynomialND (int Order, int Rank) {
			this.Rank = Rank;
			this.Order = new int[Rank];
			int Count = 1;
			for (int i = 0; i < Rank; i++) {
				Count = Count * this.Order[i];
			}
			Coefficients = new double[Count];
			PolynomialBase (this.Order, Coefficients);
		}
		public PolynomialND (int[] Order, double[] Coefficients) {
			int arraySize = 1;
			for (int i = 0; i < Order.Length; i++) {
				arraySize = arraySize * Order [i];
			}
			if (arraySize != Coefficients.Length)
				throw new Exception ("Coefficients Array must be the same size as the cumulative product of Order");
			this.Order = Order;
			this.Coefficients = Coefficients;
			this.Rank = Order.Length;
		}

		private void PolynomialBase(int[] _Order, double[] _Coefficients) {
			Rank = _Order.Length;
			Order = _Order;
			Coefficients = _Coefficients;
		}

		public int[] Order; // polynomial interpolation level in that level (1 = none, 2 = linear, 3 = quadratic, 4 = cubic, etc ..)
		public int Rank; // number of polynomial dimensions 1 = x, 2 = x-y, 3 = x-y-z, etc, 2 = L1-L2, 3 = L1-L2-L3, 4 = L1-L2-L3-L4
		public double[] Coefficients;

		public void Reshape(int Rank, int[] Order) { // use this formula to turn a polynomial as a function of x to a function of y, z, etc
			// this is most useful when performing langragian interpolation functions L(X)L(Y)L(Z)
			int TotalElements = 1;
			for (int i = 0; i < Order.Length; i++) {
				TotalElements = TotalElements * Order [i];
			}
			if (TotalElements != Coefficients.Length)
				throw new Exception ("The total number of coefficients in the polynomial must not change");
			this.Order = Order;
			this.Rank = Rank;
		}

		public static PolynomialND operator +(PolynomialND p2, PolynomialND p1) {
			int[] extent;
			if (p1.Rank != p2.Rank)
				throw new Exception ("Both polynomials need to be the same number of dimensions");
			var coefficients = addArrays(p2.Coefficients, p1.Coefficients, p1.Order, p2.Order, out extent);
			var p3 = new PolynomialND(extent, coefficients);
			return p3;
		}

		private static double[] addArrays(double[] A, double[] B, int[] ADims, int[] BDims, out int[] CDims) {
			int[] NewDims = new int[Math.Max (BDims.Length, ADims.Length)];
			for (int i = 0; i < ADims.Length; i++)
				NewDims [i] = ADims [i];
			for (int i = 0; i < BDims[i]; i++)
				NewDims [i] = Math.Max (NewDims [i], BDims [i]);
			CDims = NewDims;
			int TotalLength = 1;
			for (int i = 0; i < NewDims.Length; i++) {
				TotalLength = TotalLength * NewDims [i];
			}
			var Id = new Index (NewDims);
			var IdA = new Index (ADims);
			var IdB = new Index (BDims);
			var Coefficients = new double[TotalLength];
			for (int i = 0; i < A.Length; i++) {
				var Sub = IdA.Ind2Sub (i);
				var idx = Id.Sub2Ind(Sub);
				Coefficients [idx] = A [i];
			}
			for (int i = 0; i < B.Length; i++) {
				var Sub = IdB.Ind2Sub (i);
				var idx = Id.Sub2Ind(Sub);
				Coefficients [idx] = Coefficients [idx] + B [i];
			}
			return Coefficients;
		} 

		public static PolynomialND operator -(PolynomialND p2, PolynomialND p1) {
			return p2 + p1 * -1.0;
		}
		public static PolynomialND operator *(PolynomialND p2, PolynomialND p1) {
			var p3Extent = new int[p1.Rank];
			for (int i = 0; i < p3Extent.Length; i++) {
				p3Extent [i] = p1.Order [i] + p2.Order [i];
			}
			var p3 = new PolynomialND (p3Extent);
			var idA = new Index (p1.Order);
			var idB = new Index (p2.Order);
			var idC = new Index (p3.Order);
			int[] subC = new int[p3.Rank];
			for (int i = 0; i < p1.Coefficients.Length; i++) {
				var subA = idA.Ind2Sub (i);
				for (int j = 0; j < p2.Coefficients.Length; j++) {
					var subB = idB.Ind2Sub (j);
					for (int k = 0; k < subC.Length; k++) {
						subC [k] = subA [k] + subB [k];
					}
					int idx = idC.Sub2Ind (subC);
					p3.Coefficients [idx] = p1.Coefficients [i] * p2.Coefficients [j];
				}
			}
			return p3;
		}
		public static PolynomialND operator *(PolynomialND p1, double Scalar){
			var p2 = new PolynomialND (p1.Order);
			for (int i = 0; i < p2.Coefficients.Length; i++) {
				p2.Coefficients [i] = p1.Coefficients [i] * Scalar;
			}
			return p2;
		}
		public static PolynomialND operator *(double Scalar, PolynomialND p1) {
			return p1 * Scalar;
		}
		public double Evaluate(double[] Value) { // evaluate an ND-Polynomial
			double Ans = 0;
			double AnsLocal;
			int[] Sub;
			var Id = new Index (Order);
			for (int i = 0; i < Coefficients.Length; i++) {
				AnsLocal = Coefficients[i];
				Sub = Id.Ind2Sub (i);
				for (int j = 0; j < Value.Length; j++) {
					AnsLocal = AnsLocal * Math.Pow(Value[j],Sub[j]);
				}
				Ans = Ans + AnsLocal;
			}
			return Ans;
		}
		public PolynomialND Evaluate(double Val, int Dim) // turn a ND Polynomial into a (N-1)D one 
		{
			var OrderNew = (int [])Order.Clone ();
			OrderNew [Dim] = 1;
			int Length = Order [Dim];
			var Id = new Index (OrderNew);
			var IdOld = new Index (Order);
			var ans = new PolynomialND (OrderNew);
			for (int i = 0; i < ans.Coefficients.Length; i++) {
				double LocalCoeff = 0;
				var sub = Id.Ind2Sub (i);
				int[] subAns = (int[])sub.Clone ();
				for (int j = 0; j < Length; j++) {
					int idx = IdOld.Sub2Ind (subAns);
					LocalCoeff = LocalCoeff + Math.Pow (Val, j) * Coefficients [idx];
					subAns[Dim]++;
				}
				ans.Coefficients [i] = LocalCoeff;
			}
			return ans;

		}
		public double Integrate (Point B, Point A) {
			var pInt = Integrate (0).Integrate (1).Integrate (2);
			double Upper = pInt.Evaluate (new double[] { B.x, B.y, B.z });
			double Lower = pInt.Evaluate (new double[] { A.x, A.y, A.z });
			return Upper - Lower;
		}
		public PolynomialND Integrate (double b, double a, int dim) {
			var pInt = Integrate (dim);
			return pInt.Evaluate (b, dim) - pInt.Evaluate (a, dim);
		}
		public PolynomialND Integrate(double Val, int dim) {
			var pInt = Integrate (dim);
			return pInt.Evaluate (Val, dim);
		}
		public PolynomialND Integrate(int Dim) {
			int[] OrderNew = (int[])Order.Clone ();
			OrderNew [Dim]++; 
			var ans = new PolynomialND (OrderNew);
			var Id_Old = new Index(Order);
			var Id = new Index (ans.Order);
			for (int i = 0; i < ans.Coefficients.Length; i++) {
				var subAns = Id.Ind2Sub (i);
				subAns [Dim]--;
				int idx = Id_Old.Sub2Ind (subAns);
				ans.Coefficients [i] = Coefficients [idx] *(1.0d /  (double)subAns [Dim]);
			}
			return ans;
		}
		public PolynomialND Differentiate(int Dim) {
			int[] OrderNew = (int[])Order.Clone ();
			OrderNew [Dim]--; 
			var ans = new PolynomialND (OrderNew);
			var Id_Old = new Index(Order);
			var Id = new Index (ans.Order);
			for (int i = 0; i < ans.Coefficients.Length; i++) {
				var subAns = Id.Ind2Sub (i);
				subAns [Dim]++;
				int idx = Id_Old.Sub2Ind (subAns);
				ans.Coefficients [i] = Coefficients [idx] * subAns [Dim];
			}
			return ans;
		}
	}
}
