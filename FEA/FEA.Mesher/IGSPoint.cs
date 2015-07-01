using System;

namespace FEA.Mesher.IGES
{
    public class IGSPoint
    {
        public IGSPoint(double[] Parameters)
        {
            x = Parameters[1];
            y = Parameters[2];
            z = Parameters[3];
        }
        public double x;
        public double y;
        public double z;
    }
}

