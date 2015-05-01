using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FEA
{
    public class Node
    {
        long id;
        Point Location;
    }
    public struct Point { // copy some stuff from managed cuda
        public double x;
        public double y;
        public double z;
    };
}
