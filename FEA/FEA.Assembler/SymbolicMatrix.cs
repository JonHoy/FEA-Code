using System;

namespace FEA.Assembler
{
	public class SymbolicMatrix
	{
		public SymbolicMatrix (int Rows, int Cols, string VarName = "a")
		{
			Data = new SymbolicExpression[Rows, Cols];
			for (int i = 0; i < Rows; i++) {
				for (int j = 0; j < Cols; j++) {
					Data [i, j] = new SymbolicExpression(VarName + "_" + i.ToString () + "_" + j.ToString ());
				}
			}
			Multiplier = new SymbolicExpression("1");
		}

		SymbolicExpression[,] Data; // 
		SymbolicExpression Multiplier;
	}

	public class SymbolicExpression {
		public SymbolicExpression(string Expression) {
			this.Expression = Expression;
			TrimParentheses (); // remove excess parentheses if they exist
		}
		public static SymbolicExpression operator + (SymbolicExpression A, SymbolicExpression B) {
			var C = A.Expression + " + (" + B.Expression + ")";
			return new SymbolicExpression(C);
		}
		public static SymbolicExpression operator - (SymbolicExpression A, SymbolicExpression B) {
			var C = A.Expression + " - (" + B.Expression + ")";
			return new SymbolicExpression(C);
		}

		public static SymbolicExpression operator * (SymbolicExpression A, SymbolicExpression B) {
			var C = "(" + A.Expression + ") * (" + B.Expression + ")";
			return new SymbolicExpression(C);
		}
		public static SymbolicExpression operator / (SymbolicExpression A, SymbolicExpression B) {
			var C = "(" + A.Expression + ") / (" + B.Expression + ")";
			return new SymbolicExpression(C);
		}
		public static SymbolicExpression operator ^ (SymbolicExpression A, int n) {
			string C = A.Expression;
			var newExp = new SymbolicExpression ("C");
			if (n == 0)
				return new SymbolicExpression ("1");
			else if (n < 0)
				return (A.Reciprocal() ^ -n);
			else {
				for (int i = 1; i < n; i++) {
					newExp = newExp * A;
				}
				return newExp;
			}
		}
		public static SymbolicExpression operator ^ (SymbolicExpression A, SymbolicExpression B) {
			var C = "(" + A.Expression + ") ^ (" + B.Expression + ")";
			return new SymbolicExpression (C);
		}

		public SymbolicExpression Reciprocal() {
			var newExp = "1 / (" + Expression + ")";
			return new SymbolicExpression (newExp);
		}

		private void TrimParentheses() {
			int LeftCount = 0;
			int RightCount = 0;
			int LeftPos = 0;
			int RightPos = 0;
			// look for the first double parentheses
			for (int iL = 0; iL < Expression.Length; iL++) {
				if (Expression [iL].ToString() == "(")
					LeftCount++;
				if (iL != (Expression.Length - 1) && (Expression [iL] == Expression [iL + 1])) {
					LeftPos = (iL + 1); 
					break;
				}
				if (Expression [iL].ToString() == ")")
					LeftCount--;
			}
			for (int iR = (Expression.Length - 1); iR >= 0; iR--) {
				if (Expression [iR].ToString () == ")")
					RightCount++;  // increase nest level
				if (iR != 0 && Expression [iR] == Expression [iR - 1]) {
					RightPos = (iR - 1); 
					break;
				}
				if (Expression [iR].ToString () == "(") // Reduce current nest level
					RightCount--;
			}
			int Len = RightPos - LeftPos;
			// make sure the left parentheses and right parentheses are on the same nest count
			if (Len > 0 && LeftCount > 0 && LeftCount == RightCount) {
				Expression = Expression.Substring (LeftPos, Len + 1); 
				TrimParentheses (); // try trimming again
			}
		}

		public string Expression { get; private set; }
	}
}

