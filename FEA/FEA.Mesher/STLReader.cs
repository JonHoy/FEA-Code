using System;
using System.IO;
using ManagedCuda.VectorTypes;

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
        }
        public void SplitPart(out STLReader Part1, out STLReader Part2) {
            
        }
        uint TriangleCount;
        Triangle[] Triangles;
        double3[] NormalVector;
    }
}

