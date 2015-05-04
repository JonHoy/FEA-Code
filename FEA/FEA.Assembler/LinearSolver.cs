using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.CudaBlas;
using ManagedCuda.CudaSparse;

namespace FEA
{
    public class LinearSolver
    {
        public static float[] GPU_Single_ConjugateGradient(int[] h_col, int[] h_row, float[] h_val, float[] h_r, float[] h_x)
        {
            float tol = 1e-8f;
            int max_iter = 10000;
            var Ctx = new CudaContext();
            var cublasHandle = new ManagedCuda.CudaBlas.CudaBlasHandle();
            var cusparseHandle = new cusparseContext();
            var descr = new CudaSparseMatrixDescriptor();

            CudaDeviceVariable<int> d_col;
            CudaDeviceVariable<int> d_row;
            CudaDeviceVariable<float> d_val;
            CudaDeviceVariable<float> d_r;
            CudaDeviceVariable<float> d_alpha;
            CudaDeviceVariable<float> d_alpham1;
            CudaDeviceVariable<float> d_beta;
            CudaDeviceVariable<float> d_r1;
            CudaDeviceVariable<float> d_r0;
            CudaDeviceVariable<float> d_b;
            CudaDeviceVariable<float> d_a;
            CudaDeviceVariable<float> d_dot;

            var alpha = new float[] { 1.0f };
            var alpham1 = new float[] { -1.0f };
            var beta = new float[] { 0.0f };
            var r1 = new float[] {0.0f};
            float[] r0 = new float[] { 0.0f };
            float[] dot = new float[] { 0.0f };
            float[] b = new float[] { 0.0f };
            float[] a = new float[] { 0.0f };

            d_alpha = alpha;
            d_alpham1 = alpham1;
            d_beta = beta;
            d_r1 = r1;
            d_dot = dot;
            d_r0 = r0;
            d_b = b;

            d_col = h_col;
            d_row = h_row;
            d_val = h_val;
            CudaDeviceVariable<float> d_x = h_x;
            int N = h_x.Length;
            var rhs = new float[N];
            for (int i = 0; i < N; i++)
			{
			    rhs[i] = 1.0f;
			}
            int nz = h_val.Length;
            var d_Ax = new CudaDeviceVariable<float>(N);
            d_r = rhs;
            var d_p = new CudaDeviceVariable<float>(N);

            var State1 = CudaSparseNativeMethods.cusparseScsrmv(cusparseHandle, cusparseOperation.NonTranspose, N, N, nz, d_alpha.DevicePointer, descr.Descriptor,
                d_val.DevicePointer, d_row.DevicePointer, d_col.DevicePointer, d_x.DevicePointer, d_beta.DevicePointer, d_Ax.DevicePointer);
            
            var State2 = CudaBlasNativeMethods.cublasSaxpy_v2(cublasHandle, N, d_alpham1.DevicePointer, d_Ax.DevicePointer, 1, d_r.DevicePointer, 1);
            var State3 = CudaBlasNativeMethods.cublasSdot_v2(cublasHandle, N, d_r.DevicePointer, 1, d_r.DevicePointer, 1, d_r1.DevicePointer);

            r1 = d_r1;
            int k = 1;

            while (r1[0] > tol*tol && k <= max_iter)
            {
                if (k > 1)
                {
                    b[0] = r1[0] / r0[0];
                    d_b = b;
                    var Status = CudaBlasNativeMethods.cublasSscal_v2(cublasHandle, N, d_b.DevicePointer, d_p.DevicePointer, 1);
                    Status = CudaBlasNativeMethods.cublasSaxpy_v2(cublasHandle, N, d_alpha.DevicePointer, d_r.DevicePointer, 1, d_p.DevicePointer, 1);
                }
                else
                {
                    var Status = CudaBlasNativeMethods.cublasScopy_v2(cublasHandle, N, d_r.DevicePointer, 1, d_p.DevicePointer, 1);
                }
                CudaSparseNativeMethods.cusparseScsrmv(cusparseHandle, cusparseOperation.NonTranspose, N, N, nz, d_alpha.DevicePointer, descr.Descriptor, d_val.DevicePointer,
                    d_row.DevicePointer, d_col.DevicePointer, d_p.DevicePointer, d_beta.DevicePointer, d_Ax.DevicePointer);

                var Status2 = CudaBlasNativeMethods.cublasSdot_v2(cublasHandle, N, d_p.DevicePointer, 1, d_Ax.DevicePointer, 1, d_dot.DevicePointer);
                r1 = d_r1; dot = d_dot;
                
                a[0] = r1[0] / dot[0];
                d_a = a;
                Status2 = CudaBlasNativeMethods.cublasSaxpy_v2(cublasHandle, N, d_a.DevicePointer, d_p.DevicePointer, 1, d_x.DevicePointer, 1);
                a = d_a;
                a[0] = a[0] * -1;
                d_a = a;
                Status2 = CudaBlasNativeMethods.cublasSaxpy_v2(cublasHandle, N, d_a.DevicePointer, d_Ax.DevicePointer, 1, d_r.DevicePointer, 1);

                r0 = r1;
                d_r0 = r0;
                Status2 = CudaBlasNativeMethods.cublasSdot_v2(cublasHandle, N, d_r.DevicePointer, 1, d_r.DevicePointer, 1, d_r1.DevicePointer);
                Ctx.Synchronize();
                Console.WriteLine("Iteration = {0}, Residual = {1}",k,Math.Sqrt(r1[0]));
                k++;
            }
            h_x = d_x;
            return h_x;
        }
    };
}
