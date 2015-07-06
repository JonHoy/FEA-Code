using System;
using System.IO;
using ManagedCuda.VectorTypes;
using System.Collections.Generic;

namespace FEA.Mesher
{
    public class STLReader
    {
        // File format information cf. wikipedia
        // https://en.wikipedia.org/wiki/STL_(file_format)

        // Advantages of STL

        // 1.) Model can be easily verfied as "watertight" aka geometrically closed

        // 2.) Format is simple to read

        // 3.) Easily parallizable on the GPU

        // Disadvantages of STL

        // 1.) Exact reproduction of smooth curved surfaces is not possible 

        // 2.) Many Triangles might be required to desribe a curved surface

        // 3.) Might not handle assemblies well (there could be mesh overlap from discretization of the geometry

        // Proposed Solution:

        // Blend the watertightness and simplcity of the triangular surface mesh with NURBS
        // Essentially, fit a NURBS surface to a set of connected triangles which have small angles between their unit normals
        // Bound the nurb so that it it is geometrically water tight 
        // Alternatively, a second or third order 3d interpolation scheme could be used. where the weights are based on the inverse of the distance to the other nodes

        public STLReader(Triangle[] _Triangles, double3[] _NormalVector) {
            Triangles = _Triangles;
            NormalVector = _NormalVector;
            TriangleCount = (uint)Triangles.Length;
            Extrema = Bounds();
        }
        public STLReader(string Filename) // reads binary stl files
        {
            using (var reader = new BinaryReader(File.OpenRead(Filename)))
            {
                var header = new byte[80];
                reader.Read(header, 0, 80);
                //Header = new string(header);
                var TriCount = new byte[4];
                reader.Read(TriCount, 0, 4);
                TriangleCount = BitConverter.ToUInt32(TriCount, 0);
                Triangles = new Triangle[TriangleCount];
                NormalVector = new double3[TriangleCount];

                var TriBuffer = new byte[2 + 12 * 4]; 
                var float3Buffer = new float[TriBuffer.Length / 4];

                var A = new double3();
                var B = new double3();
                var C = new double3();
                for (int i = 0; i < TriangleCount; i++)
                {
                    reader.Read(TriBuffer, 0, TriBuffer.Length);
                    NormalVector[i] = new double3();
                    for (int j = 0; j < float3Buffer.Length; j++) {
                        float3Buffer[j] = BitConverter.ToSingle(TriBuffer, j * 4);
                    }
                    NormalVector[i].x = (double)float3Buffer[0];
                    NormalVector[i].y = (double)float3Buffer[1];
                    NormalVector[i].z = (double)float3Buffer[2];

                    A.x = (double) float3Buffer[3];
                    A.y = (double) float3Buffer[4];
                    A.z = (double) float3Buffer[5];

                    B.x = (double) float3Buffer[6];
                    B.y = (double) float3Buffer[7];
                    B.z = (double) float3Buffer[8];

                    C.x = (double) float3Buffer[9];
                    C.y = (double) float3Buffer[10];
                    C.z = (double) float3Buffer[11];

                    Triangles[i] = new Triangle(A, B, C);

                }
            }
            Extrema = Bounds();
        }
        public Plane SplittingPlane(int Dim) {
            var BoundBox = Extrema;
            Plane Slice;
            double Val;
            if (Dim == 0)
            {
                Val = BoundBox.Max.x + BoundBox.Min.x; 
            }
            else if (Dim == 1)
            {
                Val = BoundBox.Max.y + BoundBox.Min.y;
            }
            else if (Dim == 2)
            {
                Val = BoundBox.Max.z + BoundBox.Min.z;
            }
            else
            {
                throw new Exception("Dim must be 0, 1, or 2");
            }
            Val = Val / 2;
            Slice = new Plane(Val, Dim);
            return Slice;
        }

        public void SplitPart(Plane Slice, out STLReader Part1, out STLReader Part2) {
            // step 1 Divide mesh into three regions (Above Plane, Below Plane, Intersecting Plane)
            // step 2 (Triangles in the above plane region go to Part1, below plane region goes to Part2)
            // step 3 (Sub-Divide triangles that are in both planes such that the new sub-triangles exist only within Part1 or Part2)
            var Triangles1 = new List<Triangle>((int)TriangleCount/2);
            var Triangles2 = new List<Triangle>((int)TriangleCount/2);
            var Shared = new List<Triangle>();
            var NormalVector1 = new List<double3>((int)TriangleCount/2);
            var NormalVector2 = new List<double3>((int)TriangleCount/2);
            var NormalShared = new List<double3>();
            for (int i = 0; i < TriangleCount; i++)
            {
                var Loc = Triangles[i].AboveOrBelow(Slice);
                if (Loc == Location.Above)
                {
                    Triangles1.Add(Triangles[i]);
                    NormalVector1.Add(NormalVector[i]);
                }
                else if (Loc == Location.Below)
                {
                    Triangles2.Add(Triangles[i]);
                    NormalVector2.Add(NormalVector[i]);
                }
                else {
                    Shared.Add(Triangles[i]);
                    NormalShared.Add(NormalVector[i]);
                }     
            }

            for (int k = 0; k < Shared.Count; k++) 
            {
                var Tri = Shared[k];
                var AboveTris = new List<Triangle>();
                var BelowTris = new List<Triangle>();

                if (Tri.InPlane(Slice) == false) {
                    Tri.Split(Slice, out AboveTris, out BelowTris);
                }
                else
                {
                    // TODO determine solid side + or -
                    // if all the points of the triangle are in the plane, we must determine which side is the solid side,
                    // the solid side keeps the triangle, the other side loses it.
                }

                Triangles1.AddRange(AboveTris);
                for (int j = 0; j < AboveTris.Count; j++) {
                    NormalVector1.Add(NormalShared[k]);
                }
                for (int j = 0; j < BelowTris.Count; j++) {
                    NormalVector2.Add(NormalShared[k]);
                }
                Triangles2.AddRange(BelowTris);
            }

            // TODO Make a watertight triangulation on the splitting region of a solid domain
            // because we have split this part into two pieces, we must patch up the slice plane to make it watertight
            // this involves an in plane 2d delaunay triangulation. To do this, we must transform to 2d and then back to 3d 

            Part1 = new STLReader(Triangles1.ToArray(), NormalVector1.ToArray());
            Part2 = new STLReader(Triangles2.ToArray(), NormalVector2.ToArray());
        }

        public void SplitPart(string Part1FileName, string Part2FileName, int Dim = 0) {
            var Slice = SplittingPlane(Dim);
            STLReader Part1;
            STLReader Part2;
            SplitPart(Slice, out Part1, out Part2);
            Part1.WriteToFile(Part1FileName);
            Part2.WriteToFile(Part2FileName);
        }

        public bool CheckWaterTightness() {
            // for an stl surface mesh to be watertight the following conditions must be satisfied:
            // The following is Taken from wikipedia https://en.wikipedia.org/wiki/STL_(file_format)
            /*
            To properly form a 3D volume, the surface represented by any STL files must be closed and connected, 
            where every edge is part of exactly two triangles, and not self-intersecting.
            Keep in mind "internal cavities" can exist inside a part. As such they are cutoff from other surface mesh triangles
            */ 

            // To do this, we will construct a dictionary of edges to keep track of the number of times that edge is used
            // since an edge consists of 2 points 

            // TODO implement check for triangle overlap

            var EdgeCount = new Dictionary<Line, int>(); 

            for (int i = 0; i < TriangleCount; i++)
            {
                var Line1 = new Line(Triangles[i].A, Triangles[i].B);
                var Line2 = new Line(Triangles[i].B, Triangles[i].C);
                var Line3 = new Line(Triangles[i].C, Triangles[i].A);
                try
                {
                    KeyHelper(EdgeCount, Line1);
                    KeyHelper(EdgeCount, Line2);
                    KeyHelper(EdgeCount, Line3);
                }
                catch (Exception ex)
                {
                    return false;
                }

            }
            return true;
        }

        private void KeyHelper(Dictionary<Line, int> EdgeCount, Line Edge) {
            if (EdgeCount.ContainsKey(Edge))
            {
                int CurrentCount = EdgeCount[Edge];
                if (CurrentCount > 1)
                {
                    throw new Exception("Each edge must be referenced exactly 2 times!");
                }
                else
                {
                    EdgeCount[Edge] = 2;
                }
            }
            else
            {
                EdgeCount.Add(Edge, 1);
            }
        }

        public bool InsideOrOutside(double3 Pt) {
        // determine if the point is inside or outside the geometry. If the point is on the surface,
        // it will be considered inside
        // TODO implement this via point in polygon
            var Intersections = new List<double>();
            var O = Extrema.Min - 1.0; // using point in polygon we use a point that we know is outside the geometry
            // then we cast a ray
            var D = (Pt - O) + 1;
            for (int i = 0; i < Triangles.Length; i++)
            {
                var t = Triangles[i].Intersection(O, D);
                if (t != double.NaN)
                {
                    Intersections.Add(t);
                }
            }
            var results = Intersections.ToArray();
            Array.Sort(results);
            for (int i = 0; i < results.Length; i++)
            {
                if (results[i + 1] > 1)
                {
                    if (i % 2 == 0)
                        return true; // point is inside
                    else
                        return false;
                }
                    
            }
            return false;
        }


        public void WriteToFile(string Filename) {
            File.Delete(Filename);
            using (var writer = new BinaryWriter(File.OpenWrite(Filename))) {
                var header = new byte[80];
                writer.Write(header);
                writer.Write(TriangleCount);
                var TriBuffer = new byte[2 + 12 * 4];
                var Floats = new float[12];
                for (int i = 0; i < TriangleCount; i++)
                {
                    Floats[0] = (float)NormalVector[i].x;
                    Floats[1] = (float)NormalVector[i].y;
                    Floats[2] = (float)NormalVector[i].z;
                    Floats[3] = (float)Triangles[i].A.x;
                    Floats[4] = (float)Triangles[i].A.y;
                    Floats[5] = (float)Triangles[i].A.z;
                    Floats[6] = (float)Triangles[i].B.x;
                    Floats[7] = (float)Triangles[i].B.y;
                    Floats[8] = (float)Triangles[i].B.z;
                    Floats[9] = (float)Triangles[i].C.x;
                    Floats[10] = (float)Triangles[i].C.y;
                    Floats[11] = (float)Triangles[i].C.z;
                    Buffer.BlockCopy(Floats, 0, TriBuffer, 0, 48);
                    writer.Write(TriBuffer);
                }
                writer.Close();
            }
        }

        private BoundingBox Bounds() {
            
            var Min = new double3(double.MaxValue);
            var Max = new double3(double.MinValue);
            for (int i = 0; i < TriangleCount; i++)
            {
                Max = MaxHelper(Max, Triangles[i].A);
                Max = MaxHelper(Max, Triangles[i].B);
                Max = MaxHelper(Max, Triangles[i].C);

                Min = MinHelper(Min, Triangles[i].A);
                Min = MinHelper(Min, Triangles[i].B);
                Min = MinHelper(Max, Triangles[i].C);
            }
            return new BoundingBox(Min, Max);
        }

        public BoundingBox Extrema; 

        private double3 MaxHelper(double3 Val1, double3 Val2) {
            var Val = new double3();
            Val.x = Math.Max(Val1.x, Val2.x);
            Val.y = Math.Max(Val1.y, Val2.y);
            Val.z = Math.Max(Val1.z, Val2.z);
            return Val;
        }

        private double3 MinHelper(double3 Val1, double3 Val2) {
            var Val = new double3();
            Val.x = Math.Min(Val1.x, Val2.x);
            Val.y = Math.Min(Val1.y, Val2.y);
            Val.z = Math.Min(Val1.z, Val2.z);
            return Val;
        }

        uint TriangleCount;
        Triangle[] Triangles;
        double3[] NormalVector;
    }
}

