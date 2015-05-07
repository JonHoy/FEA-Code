using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FEA
{
    public class Element
    {
		public int[] Nodes; // global id of the nodes
		public int[] ConnectedElements; // id of other connected elements
		ElementFamily ElementType; // what type of element is it?
	};

	public enum ElementFamily { // Specifies base class of the element 
		Tetrahedron,
		Hexahedron
	};
}
