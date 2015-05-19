using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FEA.Assembler
{
	public struct Node
    {
        long id;
        Point Location;
    }
	public struct Point { // copy some stuff from managed cuda
        public double x;
        public double y;
        public double z;
		public static Point Cross(Point A, Point B) {
			var C = new Point();
			C.x = A.y * B.z - A.z * B.y;
			C.y = A.z * B.x - A.x * B.z;
			C.z = A.x * B.y - A.y * B.z;
			return C;
		}
		public static double Dot(Point A, Point B) {
			return A.x * B.x + A.y * B.y + A.z * B.z;
		}
		public static Point operator - (Point A, Point B) { 
			var C = new Point ();
			C.x = A.x - B.x;
			C.y = A.y - B.y;
			C.z = A.z - B.z;
			return C;
		}
		public static Point operator * (Point A, double scalar) {
			var ans = new Point ();
			ans.x = A.x * scalar;
			ans.y = A.y * scalar;
			ans.z = A.z * scalar;
			return ans;
		}
		public Point Normalize() {
			var Ans = new Point ();
			var Magnitude = Math.Sqrt (x * x + y * y + z * z);
			Ans.x = Ans.x / Magnitude;
			Ans.y = Ans.y / Magnitude;
			Ans.z = Ans.z / Magnitude;
			return Ans;
		}
    };
}
