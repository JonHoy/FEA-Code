using System;

namespace FEA.Assembler
{
	public class Index // this is a zero based row major index
	{
		public readonly int[] extent;
		private int[] cumprod;
		public readonly int Rank;
		public Index (int[] extent)
		{
			this.extent = extent;
			Rank = extent.Length;
			cumprod = new int[extent.Length];
			cumprod [cumprod.Length - 1] = 1;
			for (int i = cumprod.Length - 2; i >= 0; i--) {
				cumprod [i] = cumprod [i + 1] * extent [i + 1]; 
			}
		}
		public int[] Ind2Sub(int idx) {
			var sub = new int[extent.Length];
			for (int i = 0; i < sub.Length; i++) {
				sub [i] = idx / cumprod [i];
				idx = idx % cumprod [i];
			}
			return sub;
		}
		public int Sub2Ind(int [] Sub) {
			int idx = 0;
			for (int i = 0; i < Sub.Length; i++) {
				idx = idx + Sub [i] * cumprod [i];
			}
			return idx;
		}

	}
}

