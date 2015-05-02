using System;
using System.Collections.Generic;
using System.Text;
using ManagedCuda.BasicTypes;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace ManagedCuda.Dispatched
{
    /// <summary>
    /// Number of channels in array
    /// </summary>
    public enum CudaArray1DNumChannels
    {
        /// <summary>
        /// One channel, e.g. float1, int1, float, int
        /// </summary>
        One = 1,
        /// <summary>
        /// Two channels, e.g. float2, int2
        /// </summary>
        Two = 2,
        /// <summary>
        /// Four channels, e.g. float4, int4
        /// </summary>
        Four = 4
    }

    /// <summary>
    /// An one dimensional CUDA array
    /// </summary>
    [Obsolete("CUDA 4.0 API calls are now thread safe. Dispatched... wrapper classes are of no use now.")]
    public class DispatchedCudaArray1D : DispatchedCudaBaseClass, IDisposable
    {
        private CUDAArrayDescriptor _arrayDescriptor;
        private CUarray _cuArray;
        private CUResult res;
        private bool disposed;

        #region Constructors
        /// <summary>
        /// Creates a new CUDA array. 
        /// </summary>
        /// <param name="format"></param>
        /// <param name="size"></param>
        /// <param name="numChannels"></param>
        /// <param name="dispatcher"></param>
        public DispatchedCudaArray1D(CUArrayFormat format, SizeT size, CudaArray1DNumChannels numChannels, System.Windows.Threading.Dispatcher dispatcher)
            : base(dispatcher)
        {
            _arrayDescriptor = new CUDAArrayDescriptor();
            _arrayDescriptor.Format = format;
            _arrayDescriptor.Height = 1;
            _arrayDescriptor.Width = size;
            _arrayDescriptor.NumChannels = (uint)numChannels;

            _cuArray = new CUarray();

            object[] paramArrayCreate = { _cuArray, _arrayDescriptor };

            res = (CUResult)_dispatcher.Invoke(new cuArrayCreate_v2(DriverAPINativeMethods.ArrayManagement.cuArrayCreate_v2), paramArrayCreate);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuArrayCreate", res));
            //res = DriverAPINativeMethods.ArrayManagement.cuArrayCreate(ref _cuArray, ref _arrayDescriptor);
            if (res != CUResult.Success) throw new CudaException(res);

            _cuArray = (CUarray)paramArrayCreate[0];
            _arrayDescriptor = (CUDAArrayDescriptor)paramArrayCreate[1];
        }

        /// <summary>
        /// Creates a new CUDA array from an existing CUarray. 
        /// </summary>
        /// <param name="cuArray"></param>
        /// <param name="dispatcher"></param>
        public DispatchedCudaArray1D(CUarray cuArray, System.Windows.Threading.Dispatcher dispatcher)
            : base(dispatcher)
        {
            _cuArray = cuArray;
            _arrayDescriptor = new CUDAArrayDescriptor();

            object[] paramArrayGetDescriptor = { _arrayDescriptor, _cuArray };

            res = (CUResult)_dispatcher.Invoke(new cuArrayGetDescriptor_v2(DriverAPINativeMethods.ArrayManagement.cuArrayGetDescriptor_v2), paramArrayGetDescriptor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuArrayGetDescriptor", res));
            //res = DriverAPINativeMethods.ArrayManagement.cuArrayGetDescriptor(ref _arrayDescriptor, _cuArray);
            if (res != CUResult.Success) throw new CudaException(res);

            _arrayDescriptor = (CUDAArrayDescriptor)paramArrayGetDescriptor[0];
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~DispatchedCudaArray1D()
        {
            Dispose(false);
        }
        #endregion

        #region Dispose
        /// <summary>
        /// Dispose
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// For IDisposable
        /// </summary>
        /// <param name="fDisposing"></param>
        protected virtual void Dispose(bool fDisposing)
        {
            if (fDisposing && !disposed)
            {
                if (_dispatcher != null)
                    _dispatcher.Invoke(new cuArrayDestroy(DriverAPINativeMethods.ArrayManagement.cuArrayDestroy), _cuArray);
                disposed = true;
                //DriverAPINativeMethods.ArrayManagement.cuArrayDestroy(_cuArray);
            }
        }
        #endregion

        #region Methods
        #region CopyFromHostToArray
        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <typeparam name="T">T must be of value type, i.e. a struct</typeparam>
        /// <param name="source">source pointer to host memory</param>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        public void CopyFromHostToArray1D<T>(T[] source, SizeT offsetInBytes) where T : struct
        {
            SizeT sizeInBytes = (source.LongLength * Marshal.SizeOf(typeof(T)));
            GCHandle handle = GCHandle.Alloc(source, GCHandleType.Pinned);
            IntPtr ptr = handle.AddrOfPinnedObject();
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAIntPtr(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, ptr, sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, ptr, sizeInBytes);
            handle.Free();
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <typeparam name="T">T must be of value type, i.e. a struct</typeparam>
        /// <param name="source">source pointer to host memory</param>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        public void CopyFromHostToArray1D<T>(T source, SizeT offsetInBytes) where T : struct
        {
            SizeT sizeInBytes = Marshal.SizeOf(typeof(T));
            GCHandle handle = GCHandle.Alloc(source, GCHandleType.Pinned);
            IntPtr ptr = handle.AddrOfPinnedObject();
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAIntPtr(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, ptr, sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, ptr, sizeInBytes);
            handle.Free();
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="source">Pointer to source data</param>
        /// <param name="sizeInBytes">Number of bytes to copy</param>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        public void CopyFromHostToArray1D(IntPtr source, SizeT sizeInBytes, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAIntPtr(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, sizeInBytes);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(byte[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAByteA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * sizeof(byte)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (uint)source.LongLength);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(double[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoADoubleA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * sizeof(double)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * sizeof(double)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(float[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAFloatA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * sizeof(float)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * sizeof(float)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(int[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAIntA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * sizeof(int)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * sizeof(int)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(long[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoALongA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * sizeof(long)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * sizeof(long)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(sbyte[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoASByteA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * sizeof(sbyte)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * sizeof(sbyte)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(short[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAShortA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * sizeof(short)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * sizeof(short)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(uint[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAUIntA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * sizeof(uint)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * sizeof(uint)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(ulong[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAULongA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * sizeof(ulong)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * sizeof(ulong)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(ushort[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAUShortA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * sizeof(ushort)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * sizeof(ushort)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        #region VectorTypesArray
        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.dim3[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoADim3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.dim3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.dim3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.char1[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAChar1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.char1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.char1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.char2[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAChar2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.char2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.char2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.char3[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAChar3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.char3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.char3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.char4[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAChar4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.char4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.char4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.uchar1[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAUChar1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.uchar2[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAUChar2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.uchar3[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAUChar3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.uchar4[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAUChar4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.short1[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAShort1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.short1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.short1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.short2[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAShort2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.short2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.short2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.short3[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAShort3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.short3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.short3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.short4[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAShort4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.short4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.short4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.ushort1[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAUShort1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.ushort2[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAUShort2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.ushort3[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAUShort3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.ushort4[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAUShort4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.int1[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAInt1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.int1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.int1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.int2[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAInt2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.int2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.int2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.int3[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAInt3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.int3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.int3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.int4[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAInt4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.int4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.int4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.uint1[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAUInt1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.uint2[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAUInt2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.uint3[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAUInt3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.uint4[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAUInt4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.long1[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoALong1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.long1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.long1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.long2[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoALong2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.long2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.long2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.long3[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoALong3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.long3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.long3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.long4[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoALong4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.long4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.long4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.ulong1[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAULong1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.ulong2[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAULong2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.ulong3[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAULong3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.ulong4[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAULong4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.float1[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAFloat1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.float1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.float1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.float2[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAFloat2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.float2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.float2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.float3[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAFloat3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.float3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.float3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.float4[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAFloat4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.float4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.float4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.double1[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoADouble1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.double1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.double1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.double2[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoADouble2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.double2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.double2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.cuDoubleComplex[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoADoubleComplexA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuDoubleComplex))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuDoubleComplex))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.cuDoubleReal[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoADoubleRealA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuDoubleReal))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuDoubleReal))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.cuFloatComplex[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAFloatComplexA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuFloatComplex))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuFloatComplex))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from host to array memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="source">source array</param>
        public void CopyFromHostToArray1D(VectorTypes.cuFloatReal[] source, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyHtoAFloatRealA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2), _cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuFloatReal))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoA_v2(_cuArray, offsetInBytes, source, (source.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuFloatReal))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        #endregion
        #endregion

        #region CopyFromArrayToHost
        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <typeparam name="T">T must be of value type, i.e. a struct</typeparam>
        /// <param name="dest">Destination pointer to host memory</param>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        public void CopyFromArray1DToHost<T>(T[] dest, SizeT offsetInBytes) where T : struct
        {
            SizeT sizeInBytes = (dest.LongLength * Marshal.SizeOf(typeof(T)));
            GCHandle handle = GCHandle.Alloc(dest, GCHandleType.Pinned);
            IntPtr ptr = handle.AddrOfPinnedObject();
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHIntPtr(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), ptr, _cuArray, offsetInBytes, sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(ptr, _cuArray, offsetInBytes, sizeInBytes);
            handle.Free();
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <typeparam name="T">T must be of value type, i.e. a struct</typeparam>
        /// <param name="dest">Destination pointer to host memory</param>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        public void CopyFromArray1DToHost<T>(T dest, SizeT offsetInBytes) where T : struct
        {
            SizeT sizeInBytes = Marshal.SizeOf(typeof(T));
            GCHandle handle = GCHandle.Alloc(dest, GCHandleType.Pinned);
            IntPtr ptr = handle.AddrOfPinnedObject();
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHIntPtr(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), ptr, _cuArray, offsetInBytes, sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(ptr, _cuArray, offsetInBytes, sizeInBytes);
            handle.Free();
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="dest">Pointer to Destination data</param>
        /// <param name="sizeInBytes">Number of bytes to copy</param>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        public void CopyFromArray1DToHost(IntPtr dest, SizeT sizeInBytes, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHIntPtr(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, sizeInBytes);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(byte[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHByteA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, dest.LongLength);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (uint)dest.LongLength);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(double[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHDoubleA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * sizeof(double)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * sizeof(double)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(float[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHFloatA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * sizeof(float)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * sizeof(float)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(int[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHIntA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * sizeof(int)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * sizeof(int)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(long[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHLongA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * sizeof(long)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * sizeof(long)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(sbyte[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHSByteA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * sizeof(sbyte)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * sizeof(sbyte)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(short[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHShortA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * sizeof(short)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * sizeof(short)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(uint[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHUIntA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * sizeof(uint)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * sizeof(uint)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(ulong[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHULongA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * sizeof(ulong)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * sizeof(ulong)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(ushort[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHUShortA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * sizeof(ushort)));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * sizeof(ushort)));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        #region VectorTypesArray
        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.dim3[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHDim3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.dim3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.dim3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.char1[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHChar1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.char1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.char1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.char2[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHChar2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.char2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.char2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.char3[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHChar3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.char3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.char3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.char4[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHChar4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.char4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.char4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.uchar1[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHUChar1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.uchar2[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHUChar2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.uchar3[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHUChar3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.uchar4[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHUChar4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uchar4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.short1[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHShort1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.short1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.short1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.short2[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHShort2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.short2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.short2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.short3[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHShort3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.short3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.short3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.short4[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHShort4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.short4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.short4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.ushort1[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHUShort1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.ushort2[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHUShort2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.ushort3[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHUShort3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.ushort4[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHUShort4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ushort4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.int1[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHInt1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.int1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.int1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.int2[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHInt2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.int2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.int2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.int3[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHInt3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.int3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.int3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.int4[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHInt4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.int4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.int4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.uint1[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHUInt1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.uint2[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHUInt2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.uint3[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHUInt3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.uint4[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHUInt4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.uint4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.long1[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHLong1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.long1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.long1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.long2[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHLong2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.long2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.long2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.long3[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHLong3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.long3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.long3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.long4[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHLong4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.long4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.long4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.ulong1[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHULong1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.ulong2[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHULong2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.ulong3[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHULong3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.ulong4[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHULong4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.ulong4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.float1[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHFloat1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.float1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.float1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.float2[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHFloat2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.float2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.float2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.float3[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHFloat3A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.float3))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.float3))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.float4[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHFloat4A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.float4))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.float4))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.double1[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHDouble1A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.double1))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.double1))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.double2[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHDouble2A(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.double2))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.double2))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.cuDoubleComplex[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHDoubleComplexA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuDoubleComplex))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuDoubleComplex))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.cuDoubleReal[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHDoubleRealA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuDoubleReal))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuDoubleReal))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.cuFloatComplex[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHFloatComplexA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuFloatComplex))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuFloatComplex))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to host memory
        /// </summary>
        /// <param name="offsetInBytes">Offset in bytes of destination array</param>
        /// <param name="dest">Destination array</param>
        public void CopyFromArray1DToHost(VectorTypes.cuFloatReal[] dest, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoHFloatRealA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2), dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuFloatReal))));
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoH", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoH_v2(dest, _cuArray, offsetInBytes, (dest.LongLength * Marshal.SizeOf(typeof(VectorTypes.cuFloatReal))));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        #endregion
        #endregion

        /// <summary>
        /// Copy data from array to array
        /// </summary>
        /// <param name="dest">Destination array</param>
        /// <param name="aBytesToCopy">Size of memory copy in bytes</param>
        /// <param name="destOffset">Offset in bytes of destination array</param>
        /// <param name="sourceOffset">Offset in bytes of source array</param>
        public void CopyFromThisToArray1D(DispatchedCudaArray1D dest, SizeT aBytesToCopy, SizeT destOffset, SizeT sourceOffset)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoA_v2), dest.CUArray, destOffset, this.CUArray, sourceOffset, aBytesToCopy);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpyAtoA(dest.CUArray, destOffset, this.CUArray, sourceOffset, aBytesToCopy);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to array
        /// </summary>
        /// <param name="source">Destination array</param>
        /// <param name="aBytesToCopy">Size of memory copy in bytes</param>
        /// <param name="destOffset">Offset in bytes of destination array</param>
        /// <param name="sourceOffset">Offset in bytes of source array</param>
        public void CopyFromArray1DToThis(DispatchedCudaArray1D source, SizeT aBytesToCopy, SizeT destOffset, SizeT sourceOffset)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoA_v2), this.CUArray, destOffset, source.CUArray, sourceOffset, aBytesToCopy);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpyAtoA(this.CUArray, destOffset, source.CUArray, sourceOffset, aBytesToCopy);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from array to device
        /// </summary>
        /// <param name="dest">DevicePointer to copy data to</param>
        /// <param name="aBytesToCopy">number of bytes to copy</param>
        /// <param name="offsetInBytes">Offset in bytes of source array</param>
        public void CopyFromArray1DToDevice(CUdeviceptr dest, SizeT aBytesToCopy, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyAtoD(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyAtoD_v2), dest, _cuArray, offsetInBytes, aBytesToCopy);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyAtoD", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpyAtoD(dest, _cuArray, offsetInBytes, aBytesToCopy);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from device to array
        /// </summary>
        /// <param name="source">DevicePointer to copy data from</param>
        /// <param name="aBytesToCopy">number of bytes to copy</param>
        /// <param name="offsetInBytes">Offset in bytes of source array</param>
        public void CopyFromDeviceToArray1D(CUdeviceptr source, SizeT aBytesToCopy, SizeT offsetInBytes)
        {
            CUResult res;
            res = (CUResult)_dispatcher.Invoke(new cuMemcpyDtoA(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoA_v2), _cuArray, offsetInBytes, source, aBytesToCopy);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoA", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy.cuMemcpyDtoA(_cuArray, offsetInBytes, source, aBytesToCopy);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        #endregion

        #region Properties
        /// <summary>
        /// Returns the array width in elements
        /// </summary>
        public SizeT Width
        {
            get { return _arrayDescriptor.Width; }
        }

        /// <summary>
        /// Returns the array width in bytes
        /// </summary>
        public SizeT WidthInBytes
        {
            get { return _arrayDescriptor.Width * _arrayDescriptor.NumChannels * CudaHelperMethods.GetChannelSize(_arrayDescriptor.Format); }
        }

        /// <summary>
        /// Returns the wrapped CUarray
        /// </summary>
        public CUarray CUArray
        {
            get { return _cuArray; }
        }

        /// <summary>
        /// Returns the wrapped CUDAArrayDescriptor
        /// </summary>
        public CUDAArrayDescriptor ArrayDescriptor
        {
            get { return _arrayDescriptor; }
        }
        #endregion
    }
}
