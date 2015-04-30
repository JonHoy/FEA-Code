using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FEA.Assembler
{
    public struct Element
    {
        public int id;
        public Node[] Nodes;
        public PolyMatrix ShapeFunction;
    }


}
