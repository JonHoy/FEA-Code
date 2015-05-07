using System;
using System.Collections.Generic;
using System.Linq;
namespace FEA
{
	public class Sparse
	{
		// this is a container for the indicies of a sparse matrix not the values
		public int[] Rows;
		public int[] RowsCompressed;
		public int[] Cols;

		public Sparse (int[] Rows, int[] Cols, int[] RowsCompressed) {
			this.Rows = Rows;
			this.Cols = Cols;
			this.RowsCompressed = RowsCompressed;
		}
		public Sparse (Element[] Elements, Point[] Points)
		{
			Initializer (false, Elements, Points);
			Initializer (true, Elements, Points);
		}
		private void Initializer(bool IsAllocated, Element[] Elements, Point[] Points) {
			if (IsAllocated == false)
				RowsCompressed = new int[Points.Length + 1];
			int MaxRowWidth;
			for (int i = 0; i < Elements.Length; i++) {
				MaxRowWidth = Elements[i].Nodes.Length;
				for (int j = 0; j < Elements[i].ConnectedElements.Length; j++) {
					int ConnectionId = Elements [i].ConnectedElements [j];
					MaxRowWidth = MaxRowWidth + Elements [ConnectionId].Nodes.Length;
				}
				var Connections = new List<int>(MaxRowWidth);
				for (int j = 0; j < Elements[i].ConnectedElements.Length; j++) {
					int ConnectionId = Elements [i].ConnectedElements [j];
					Connections.AddRange (Elements [ConnectionId].Nodes.ToList());
				}
				Connections.Distinct();
				if (IsAllocated == false) {
					RowsCompressed [i + 1] = RowsCompressed [i] + Connections.Count;
				} 
				else {
					var Block = Connections.ToArray ();
					Buffer.BlockCopy (Block, 0, Cols, RowsCompressed[i], sizeof(int) * Block.Length);
					for (int j = 0; j < Block.Length; j++) {
						Rows [RowsCompressed [i] + j] = j;
					}
				}

			}
			if (IsAllocated == false) {
				Rows = new int[RowsCompressed [RowsCompressed.Length - 1]];
				Cols = new int[Rows.Length]; 
			}
		}
		public int getIdx(int i, int j) { // return 1-D index number for the sparse 2d array 
			int Start = RowsCompressed [i];
			int End = RowsCompressed [i + 1];
			return Array.BinarySearch (Cols, Start, End - Start, j);
		}  
	}
}

