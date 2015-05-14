using System;

namespace FEA
{
	public class FiniteVolume
	{
		int[] ElementLocations; // starting point indices of a new element in the array
		int[] NodeIDS; // global ids of all the nodes in each element
		int[] NodeLocations; // starting point indices of a new node in the array
		int[] ElementIDS; // global ids of all the elements that are connected to each node
		public FiniteVolume(int[] ElementLocations, int[] NodeIDS, int NodeCount)
		{
			this.ElementLocations = ElementLocations;
			this.NodeIDS = NodeIDS;
			NodeLocations = new int[NodeCount + 1];
			for (int i = 0; i < NodeIDS.Length; i++) { // loop through each node contained in each element
				int NodeNumber = NodeIDS[i] + 1;
				NodeLocations [NodeNumber]++; // count how elements reference that particular node
			}
			for (int i = 1; i < NodeLocations.Length; i++) {
				NodeLocations [i] = NodeLocations [i - 1] + NodeLocations[i]; // sum up each 
			}
			var ConnectionCounter = new int[NodeCount];
			int[] ElementIDS = new int[NodeLocations [NodeLocations.Length - 1]];
			for (int iElement = 0; iElement < ElementLocations.Length; iElement++) {	
				int StartPt = ElementLocations [iElement];
				int EndPt = ElementLocations [iElement + 1];
				for (int iNode = StartPt; iNode < EndPt; iNode++) {
					int CurrentNode = NodeIDS [iNode];
					int StartAddress = NodeLocations [CurrentNode];
					int offset = StartAddress + ConnectionCounter [CurrentNode];
					ElementIDS[offset] = iElement;
					ConnectionCounter[CurrentNode]++;
				}
			}

		}
	}
}

