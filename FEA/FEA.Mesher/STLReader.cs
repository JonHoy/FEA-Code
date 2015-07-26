using System;
using System.IO;
using ManagedCuda.VectorTypes;
using System.Collections.Generic;
using System.Linq;
using TriangleNet.Geometry;
using System.Threading.Tasks;

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
                    var Tris = Tri.Split(Slice);
                    foreach (var item in Tris)
                    {
                        if (item.AboveOrBelow(Slice) == Location.Above)
                        {
                            AboveTris.Add(item);
                        }
                        else if (item.AboveOrBelow(Slice) == Location.Below)
                        {
                            BelowTris.Add(item);
                        }
                        else
                        {
                            throw new Exception("The triangle must be above or below but not on the plane");
                        }
                    }
                }
                else
                {
                    // TODO determine solid side + or -
                    // if all the points of the triangle are in the plane, we must determine which side is the solid side,
                    // the solid side keeps the triangle, the other side loses it.
                }


                for (int j = 0; j < AboveTris.Count; j++) {
                    NormalVector1.Add(NormalShared[k]);
                    Triangles1.Add(AboveTris[j]);
                }
                for (int j = 0; j < BelowTris.Count; j++) {
                    NormalVector2.Add(NormalShared[k]);
                    Triangles2.Add(BelowTris[j]);
                }

            }

            // TODO Make a watertight triangulation on the splitting region of a solid domain
            // because we have split this part into two pieces, we must patch up the slice plane to make it watertight
            // this involves an in plane 2d delaunay triangulation. To do this, we must transform to 2d and then back to 3d 

            ///var PatchTris = PatchSlice(Slice);

            // now we have to account for in plane triangles. when given patches, the side above the unit normal will have that triangle removed

            Part1 = new STLReader(Triangles1.ToArray(), NormalVector1.ToArray());
            Part2 = new STLReader(Triangles2.ToArray(), NormalVector2.ToArray());

    
            var Patch = Part1.PatchSlice(Slice); // since we sliced open the stl file we must patch it back up
            // TODO remove 2d line segments that are redundant and can be reduced to a single segment. This violates mesh conformity but not water tightness
            // maybe add in line segments that are too far apart as well to get a more uniform output mesh
            int Part1OldLength = Part1.Triangles.Length;
            int Part2OldLength = Part2.Triangles.Length;


            Array.Resize<Triangle>(ref Part1.Triangles, Part1OldLength + Patch.Length);
            Array.Copy(Patch, 0, Part1.Triangles, Part1OldLength, Patch.Length);
            Array.Resize<Triangle>(ref Part2.Triangles, Part2OldLength + Patch.Length);
            Array.Copy(Patch, 0, Part2.Triangles, Part2OldLength, Patch.Length);

            Array.Resize<double3>(ref Part1.NormalVector, Part1OldLength + Patch.Length);
            Array.Resize<double3>(ref Part2.NormalVector, Part2OldLength + Patch.Length);
            double3 Part1Normal = -1.0 * Slice.UnitNormal;
            double3 Part2Normal = Slice.UnitNormal;
            for (int i = 0; i < Patch.Length; i++)
            {
                Part1.NormalVector[i + Part1OldLength] = Part1Normal;
                Part2.NormalVector[i + Part2OldLength] = Part2Normal;
            }
            bool Status1 = Part1.CheckWaterTightness();
            bool Status2 = Part2.CheckWaterTightness();

            Part1.TriangleCount = (uint) Part1.Triangles.Length;
            Part2.TriangleCount = (uint) Part2.Triangles.Length;

        }

        public List<STLReader> RecursiveSplit(int MaxTriCount) {
            // this algorithm subdivides the stl part into a list of small pieces
            // This is 3d quadtree like in that it keeps on dividing along the dimension of greatest length
            // eg if the part is bounded by 12in x 5in x 16in box
            // the new boxes are 12 x 5 x 8 in size
            // and if needs be, then further subdivided to 6 x 5 x 8 then to 6 x 5 x 4 and so on...
            //MaxTriangleCount = MaxTriCount;
            var MasterList = new List<STLReader>();
            if (TriangleCount < MaxTriCount)
            {
                MasterList.Add(this);
                return MasterList;
            }

            var Lengths = new double[3];
            Lengths[0] = Extrema.Max.x - Extrema.Min.x;
            Lengths[1] = Extrema.Max.y - Extrema.Min.y;
            Lengths[2] = Extrema.Max.z - Extrema.Min.z;
            int MaxDim = 0;
            double MaxLength = 0;
            double SliceLocation = 0;
            double StartPt = 0;
            for (int i = 0; i < 3; i++)
            {
                if (Lengths[i] > MaxLength)
                {
                    MaxLength = Lengths[i];
                    MaxDim = i;
                }
            }
            if (MaxDim == 0)
            {
                SliceLocation = Extrema.Min.x + MaxLength / 2;
            }
            else if (MaxDim == 1){
                SliceLocation = Extrema.Min.y + MaxLength / 2;
            }
            else {
                SliceLocation = Extrema.Min.z + MaxLength / 2;
            }

            var Slice = new Plane(SliceLocation, MaxDim);
            STLReader Part1;
            STLReader Part2;
            SplitPart(Slice, out Part1, out Part2);
            Part1.WriteToFile("Part1_" + Part1.TriangleCount.ToString() + ".stl");
            Part2.WriteToFile("Part2_" + Part2.TriangleCount.ToString() + ".stl");
            /*
            // Multithreaded implementation
            var Obj1 = new SubDividerObj();
            Obj1.Part = Part1;
            Obj1.MaxTriCount = MaxTriCount;
            var Obj2 = new SubDividerObj();
            Obj2.Part = Part2;
            Obj2.MaxTriCount = MaxTriCount;
            var Thrd1 = new System.Threading.Thread(SubDividerObj.DoWork);
            var Thrd2 = new System.Threading.Thread(SubDividerObj.DoWork);
            Thrd1.Start(Obj1);
            Thrd2.Start(Obj2);
            Thrd1.Join();
            Thrd2.Join();
            MasterList.AddRange(Obj1.Output);
            MasterList.AddRange(Obj2.Output);
            */
            //
            // Single Threaded Implementation
            MasterList.AddRange(Part1.RecursiveSplit(MaxTriCount));
            MasterList.AddRange(Part2.RecursiveSplit(MaxTriCount));
            return MasterList;
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
            // TODO implement this via point in polygon (Need to account for cases where the ray is inside a triangle)

            // cast three orthonormal test rays to determine inside or out
            var O1 = new double3(Extrema.Max.x + 1.0, Pt.y, Pt.z);
            var O2 = new double3(Pt.x, Extrema.Max.y + 1.0, Pt.z);
            var O3 = new double3(Pt.x, Pt.y, Extrema.Max.z + 1);

            var S1 = InsideOutsideHelper(Pt, O1);
            var S2 = InsideOutsideHelper(Pt, O2);
            var S3 = InsideOutsideHelper(Pt, O3);

            int TrueCount = 0;
            TrueCount += Convert.ToInt32(S1);
            TrueCount += Convert.ToInt32(S2);
            TrueCount += Convert.ToInt32(S3);

            if (TrueCount > 1)
            {
                return true;
            }
            else 
            {
                return false;
            }
        }

        private bool InsideOutsideHelper(double3 Pt, double3 O) {
            var Intersections = new List<double>();
            var D = (Pt - O);
            int AboveCount = 0;
            int BelowCount = 0;
            for (int i = 0; i < Triangles.Length; i++)
            {
                var t = Triangles[i].Intersection(O, D);
                if (!double.IsNaN(t))
                {
                    if (t > 1)
                        AboveCount++;
                    else if (t > 0)
                        BelowCount++;
                }
            }
            if (AboveCount == BelowCount && AboveCount % 2 != 0)
                return true;
            else
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

        public static float3 ToFloat3(double3 Val) {
            var Ans = new float3((float)Val.x,
                          (float)Val.y,
                          (float)Val.z);
            return Ans;
        }

        public static double3 ToDouble3(float3 Val) {
            var Ans = new double3((double)Val.x,
                (double)Val.y,
                (double)Val.z);
            return Ans;
        }
        // TODO Generalize Slice to be more than just a plane (Maybe a parametric surface or NURB?)
        // Implement with Binary Space Partitioning
        private Triangle[] PatchSlice(Plane Slice) { // this function triangulates the slicing plane so that the split parts are watertight
            var PlanePts = new HashSet<double3>(); // these are the points in the plane
            var PlaneTris = new List<Triangle>(); // these will be holes or required triangles depending on the unit normal
            var PlaneLines = new HashSet<Line>(); // these are required edges in the surface triangulation 
            for (int i = 0; i < Triangles.Length; i++)
            {
                var Tri = Triangles[i];
                var LocA = Slice.AboveOrBelow(Tri.A);
                var LocB = Slice.AboveOrBelow(Tri.B);
                var LocC = Slice.AboveOrBelow(Tri.C);

                Tri.A = ToDouble3(ToFloat3(Tri.A));
                Tri.B = ToDouble3(ToFloat3(Tri.B));
                Tri.C = ToDouble3(ToFloat3(Tri.C));

                if (LocA == Location.On)
                    PlanePts.Add(Tri.A);
                
                    
                if (LocB == Location.On)
                    PlanePts.Add(Tri.B);

                if (LocC == Location.On)
                    PlanePts.Add(Tri.C);
                
                if (LocA == Location.On && LocB == Location.On && LocC == Location.On)
                    PlaneTris.Add(Tri);
                else if (LocA == Location.On && LocB == Location.On)
                    PlaneLines.Add(new Line(Tri.A, Tri.B));
                else if (LocB == Location.On && LocC == Location.On)
                    PlaneLines.Add(new Line(Tri.B, Tri.C));
                else if (LocA == Location.On && LocB == Location.On)
                    PlaneLines.Add(new Line(Tri.A, Tri.C));
            }

            var Pts3D = PlanePts.ToArray();

            var Lines = PlaneLines.ToArray();

            var x_new = Pts3D[1] - Pts3D[0]; 
            x_new.Normalize(); // define the new x-axis as the vector between the  first two
            // coplanar points
            var z_new = Slice.UnitNormal;

            var Poly = new Polygon(); 

            double ZOffset = Slice.Transform(Pts3D[0], x_new).z;

            for (int i = 0; i < Pts3D.Length; i++)
            {
                var Pt_new = Slice.Transform(Pts3D[i], x_new);
                var Vertex_new = new Vertex(Pt_new.x, Pt_new.y);
                Poly.Add(Vertex_new);
            }

            var EdgeSet = new HashSet<Edge>();
            for (int i = 0; i < Lines.Length; i++)
            {
                var PtA = Lines[i].A;
                var PtB = Lines[i].B;
                PtA = Slice.Transform(PtA, x_new);
                PtB = Slice.Transform(PtB, x_new);
                var P1 = new Vertex(PtA.x, PtA.y);
                var P2 = new Vertex(PtB.x, PtB.y);
                int I1 = Poly.Points.IndexOf(P1);
                int I2 = Poly.Points.IndexOf(P2);
                if (I1 == -1 || I2 == -1)
                    Console.WriteLine("Bad Segment Found!");

                var Edge = new Edge(Math.Min(I1, I2),Math.Max(I1, I2));
                EdgeSet.Add(Edge);
            }
            var PolyEdges = EdgeSet.ToArray();
            foreach (var item in PolyEdges)
            {
                Poly.Add(item);
            }
            //Poly = RemoveRedundantSegments(Poly.Segments, Poly.Points);

            bool Status = CheckWaterTightness(Poly.Segments, Poly.Points);

            TriangleNet.IO.TriangleWriter.WritePoly(Poly, "PolyTest.poly");

            if (Status == false) {
                Console.WriteLine("Warning, Inconsistent cross section detected!");
                //throw new Exception("STL File is not watertight!");
            }

            var qualityOptions = new TriangleNet.Meshing.QualityOptions();
            qualityOptions.MinimumAngle = 20;
            qualityOptions.MaximumAngle = 140;
            
            var myMesher = new TriangleNet.Meshing.GenericMesher();
            var myMesh = (TriangleNet.Mesh)myMesher.Triangulate(Poly, qualityOptions); // Poly needs to be broken up into self contained singular regions   

            // TODO: 
            // How does this fair when the cross section is multiple separate regions adhering to Jordan Curve Theorem
            // Each region/ closed curve is a linked list.   


            // FIXME: Write algorithm to break up seperate regions along the plane

            // Also account for holes being punched in the mesh (One side gets a hole, the mirror doesnt) when the cutting plane is along an interior surface

            // TODO Add back in unit normals to patch

            var Ans = new Triangle[myMesh.Triangles.Count];
            if (Ans.Length == 0)
            {
                throw new Exception("Patch Meshing Failure Detected!");
            }
            TriangleNet.IO.TriangleWriter.Write(myMesh, "MeshTest.ele");
            for (int i = 0; i < Ans.Length; i++)
            {
                var myTri = myMesh.Triangles.ElementAt(i);
                var P0 = myMesh.Vertices.ElementAt(myTri.P0);
                var P1 = myMesh.Vertices.ElementAt(myTri.P1);
                var P2 = myMesh.Vertices.ElementAt(myTri.P2);
                var PtA = new double3(P0.X, P0.Y, ZOffset);
                var PtB = new double3(P1.X, P1.Y, ZOffset);
                var PtC = new double3(P2.X, P2.Y, ZOffset);
                PtA = Slice.UnTransform(PtA, x_new); // map back to 3d
                PtB = Slice.UnTransform(PtB, x_new);
                PtC = Slice.UnTransform(PtC, x_new);
                Ans[i] = new Triangle(PtA, PtB, PtC);
            }
            return Ans;
        }

        private int IndexOf(Vertex Pt, List<Vertex> Set) {
            int ans = -1;
            for (int i = 0; i < Set.Count; i++)
            {
                if (Set[i].Equals(Pt))
                {
                    ans = i;
                    break;
                }
            }
            return ans;
        }

        private bool CheckWaterTightness(List<IEdge> Edges, List<Vertex> Points) {
            var ReferenceCount = new int[Points.Count];
            foreach (var Line in Edges)
            {
                ReferenceCount[Line.P0]++;
                ReferenceCount[Line.P1]++;
            }
            for (int i = 0; i < ReferenceCount.Length; i++)
            {
                if (ReferenceCount[i] != 2)
                {
                    return false;
                }
            }
            return true;
        }

        public uint TriangleCount;
        public Triangle[] Triangles; 
        public double3[] NormalVector;
    }

    class SubDividerObj {
        public int MaxTriCount;
        public STLReader Part;
        public List<STLReader> Output;

        public SubDividerObj() {
        }

        public static void DoWork(object data) {
            var Obj = (SubDividerObj) data;
            Obj.Output = Obj.Part.RecursiveSplit(Obj.MaxTriCount);
        }
    }

}


