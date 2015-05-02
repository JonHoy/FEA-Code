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
    public enum CudaArray3DNumChannels
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
    /// A two dimansional CUDA array
    /// </summary>
    [Obsolete("CUDA 4.0 API calls are now thread safe. Dispatched... wrapper classes are of no use now.")]
    public class DispatchedCudaArray3D : DispatchedCudaBaseClass, IDisposable
    {
        CUDAArray3DDescriptor _array3DDescriptor;
        CUarray _cuArray;
        CUResult res;
        bool disposed;

        #region Constructors
        /// <summary>
        /// Creates a new CUDA array. 
        /// </summary>
        /// <param name="format"></param>
        /// <param name="height">In elements</param>
        /// <param name="width">In elements</param>
        /// <param name="depth">In elements</param>
        /// <param name="numChannels"></param>
        /// <param name="flags"></param>
        /// <param name="dispatcher"></param>
        public DispatchedCudaArray3D(CUArrayFormat format, SizeT height, SizeT width, SizeT depth, CudaArray3DNumChannels numChannels, CUDAArray3DFlags flags, System.Windows.Threading.Dispatcher dispatcher)
            : base(dispatcher)
        {
            _array3DDescriptor = new CUDAArray3DDescriptor();
            _array3DDescriptor.Format = format;
            _array3DDescriptor.Height = height;
            _array3DDescriptor.Width = width;
            _array3DDescriptor.Depth = depth;
            _array3DDescriptor.Flags = flags;
            _array3DDescriptor.NumChannels = (uint)numChannels;
            
            _cuArray = new CUarray();

            object[] paramArrayCreate = { _cuArray, _array3DDescriptor };

            res = (CUResult)_dispatcher.Invoke(new cuArray3DCreate_v2(DriverAPINativeMethods.ArrayManagement.cuArray3DCreate_v2), paramArrayCreate);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuArray3DCreate", res));
            //res = DriverAPINativeMethods.ArrayManagement.cuArray3DCreate(ref _cuArray, ref _arrayDescriptor);
            if (res != CUResult.Success) throw new CudaException(res);

            _cuArray = (CUarray)paramArrayCreate[0];
            _array3DDescriptor = (CUDAArray3DDescriptor)paramArrayCreate[1];
        }

        /// <summary>
        /// Creates a new CUDA array from an existing CUarray. 
        /// </summary>
        /// <param name="cuArray"></param>
        /// <param name="dispatcher"></param>
        public DispatchedCudaArray3D(CUarray cuArray, System.Windows.Threading.Dispatcher dispatcher)
            : base(dispatcher)
        {
            _cuArray = cuArray;
            _array3DDescriptor = new CUDAArray3DDescriptor();

            object[] paramArrayGetDescriptor = { _array3DDescriptor, _cuArray };

            res = (CUResult)_dispatcher.Invoke(new cuArray3DGetDescriptor_v2(DriverAPINativeMethods.ArrayManagement.cuArray3DGetDescriptor_v2), paramArrayGetDescriptor);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuArray3DGetDescriptor", res));
            //res = DriverAPINativeMethods.ArrayManagement.cuArray3DGetDescriptor(ref _arrayDescriptor, _cuArray);
            if (res != CUResult.Success) throw new CudaException(res);

            _array3DDescriptor = (CUDAArray3DDescriptor)paramArrayGetDescriptor[0];
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~DispatchedCudaArray3D()
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
                //DriverAPINativeMethods.ArrayManagement.cuArrayDestroy(_cuArray);
                disposed = true;
            }
        }
        #endregion

        #region Methods
        /// <summary>
        /// A raw data copy method
        /// </summary>
        /// <param name="copyParameters">3D copy paramters</param>
        public void CopyData(CUDAMemCpy3D copyParameters)
        {
            object[] paramMemcpy3D = { copyParameters };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy3D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2), paramMemcpy3D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy3D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2(ref copyParameters);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy from Host to this array
        /// </summary>
        /// <param name="hostSrc">Source</param>
        /// <param name="elementSizeInBytes"></param>
        public void CopyFromHostToThis(IntPtr hostSrc, SizeT elementSizeInBytes)
        {
            CUDAMemCpy3D copyParams = new CUDAMemCpy3D();

            copyParams.srcHost = hostSrc;
            copyParams.srcMemoryType = CUMemoryType.Host;
            copyParams.dstArray = _cuArray;
            copyParams.dstMemoryType = CUMemoryType.Array;
            copyParams.Depth = _array3DDescriptor.Depth;
            copyParams.Height = _array3DDescriptor.Height;
            copyParams.WidthInBytes = _array3DDescriptor.Width * elementSizeInBytes;

            object[] paramMemcpy3D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy3D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2), paramMemcpy3D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy3D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2(ref copyParams);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from this array to host
        /// </summary>
        /// <param name="hostDest">IntPtr to destination in host memory</param>
        /// <param name="elementSizeInBytes"></param>
        public void CopyFromThisToHost(IntPtr hostDest, SizeT elementSizeInBytes)
        {
            CUDAMemCpy3D copyParams = new CUDAMemCpy3D();
            copyParams.dstHost = hostDest;
            copyParams.dstMemoryType = CUMemoryType.Host;
            copyParams.srcArray = _cuArray;
            copyParams.srcMemoryType = CUMemoryType.Array;
            copyParams.Depth = _array3DDescriptor.Depth;
            copyParams.Height = _array3DDescriptor.Height;
            copyParams.WidthInBytes = _array3DDescriptor.Width * elementSizeInBytes;

            object[] paramMemcpy3D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy3D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2), paramMemcpy3D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy3D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2(ref copyParams);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy from Host to this array
        /// </summary>
        /// <typeparam name="T">Host array base type</typeparam>
        /// <param name="hostSrc">Source</param>
        public void CopyFromHostToThis<T>(T[] hostSrc)
        {
            GCHandle handle = GCHandle.Alloc(hostSrc, GCHandleType.Pinned);

            CUDAMemCpy3D copyParams = new CUDAMemCpy3D();
            copyParams.srcHost = handle.AddrOfPinnedObject();
            copyParams.srcMemoryType = CUMemoryType.Host;
            copyParams.dstArray = _cuArray;
            copyParams.dstMemoryType = CUMemoryType.Array;
            copyParams.Depth = _array3DDescriptor.Depth;
            copyParams.Height = _array3DDescriptor.Height;
            copyParams.WidthInBytes = _array3DDescriptor.Width * (uint)Marshal.SizeOf(typeof(T));

            object[] paramMemcpy3D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy3D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2), paramMemcpy3D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy3D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2(ref copyParams);
            handle.Free();
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from this array to host
        /// </summary>
        /// <typeparam name="T">Host array base type</typeparam>
        /// <param name="hostDest">Destination</param>
        public void CopyFromThisToHost<T>(T[] hostDest)
        {
            GCHandle handle = GCHandle.Alloc(hostDest, GCHandleType.Pinned);

            CUDAMemCpy3D copyParams = new CUDAMemCpy3D();
            copyParams.dstHost = handle.AddrOfPinnedObject();
            copyParams.dstMemoryType = CUMemoryType.Host;
            copyParams.srcArray = _cuArray;
            copyParams.srcMemoryType = CUMemoryType.Array;
            copyParams.Depth = _array3DDescriptor.Depth;
            copyParams.Height = _array3DDescriptor.Height;
            copyParams.WidthInBytes = _array3DDescriptor.Width * (uint)Marshal.SizeOf(typeof(T));

            object[] paramMemcpy3D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy3D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2), paramMemcpy3D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy3D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2(ref copyParams);
            handle.Free();
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy from a pitched device variable to this array
        /// </summary>
        /// <param name="deviceVariable">Source</param>
        /// <param name="elementSizeInBytes"></param>
        public void CopyFromDeviceToThis(CUdeviceptr deviceVariable, SizeT elementSizeInBytes)
        {
            CopyFromDeviceToThis(deviceVariable, elementSizeInBytes, 0);
        }

        /// <summary>
        /// Copy from a pitched device variable to this array
        /// </summary>
        /// <param name="deviceVariable">Source</param>
        /// <param name="elementSizeInBytes"></param>
        /// <param name="pitch">Pitch in bytes</param>
        public void CopyFromDeviceToThis(CUdeviceptr deviceVariable, SizeT elementSizeInBytes, SizeT pitch)
        {
            CUDAMemCpy3D copyParams = new CUDAMemCpy3D();
            copyParams.srcDevice = deviceVariable;
            copyParams.srcMemoryType = CUMemoryType.Device;
            copyParams.srcPitch = pitch;
            copyParams.dstArray = _cuArray;
            copyParams.dstMemoryType = CUMemoryType.Array;
            copyParams.Depth = _array3DDescriptor.Depth;
            copyParams.Height = _array3DDescriptor.Height;
            copyParams.WidthInBytes = _array3DDescriptor.Width * elementSizeInBytes;

            object[] paramMemcpy3D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy3D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2), paramMemcpy3D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy3D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2(ref copyParams);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy from this array to a pitched device variable
        /// </summary>
        /// <param name="deviceVariable">Destination</param>
        /// <param name="elementSizeInBytes"></param>
        public void CopyFromThisToDevice(CUdeviceptr deviceVariable, SizeT elementSizeInBytes)
        {
            CopyFromThisToDevice(deviceVariable, elementSizeInBytes, 0);
        }

        /// <summary>
        /// Copy from this array to a pitched device variable
        /// </summary>
        /// <param name="deviceVariable">Destination</param>
        /// <param name="elementSizeInBytes"></param>
        /// <param name="pitch">Pitch in bytes</param>
        public void CopyFromThisToDevice(CUdeviceptr deviceVariable, SizeT elementSizeInBytes, SizeT pitch)
        {
            CUDAMemCpy3D copyParams = new CUDAMemCpy3D();
            copyParams.dstDevice = deviceVariable;
            copyParams.dstMemoryType = CUMemoryType.Device;
            copyParams.dstPitch = pitch;
            copyParams.srcArray = _cuArray;
            copyParams.srcMemoryType = CUMemoryType.Array;
            copyParams.Depth = _array3DDescriptor.Depth;
            copyParams.Height = _array3DDescriptor.Height;
            copyParams.WidthInBytes = _array3DDescriptor.Width * elementSizeInBytes;

            object[] paramMemcpy3D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy3D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2), paramMemcpy3D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy3D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2(ref copyParams);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy array to array
        /// </summary>
        /// <param name="sourceArray"></param>
        public void CopyFromArrayToThis(DispatchedCudaArray3D sourceArray)
        {
            CUDAMemCpy3D copyParams = new CUDAMemCpy3D();
            copyParams.srcArray = sourceArray.CUArray;
            copyParams.srcMemoryType = CUMemoryType.Array;
            copyParams.dstArray = _cuArray;
            copyParams.dstMemoryType = CUMemoryType.Array;
            copyParams.Depth = sourceArray._array3DDescriptor.Depth;
            copyParams.Height = sourceArray.Array3DDescriptor.Height;
            copyParams.WidthInBytes = sourceArray.Array3DDescriptor.Width * CudaHelperMethods.GetChannelSize(sourceArray.Array3DDescriptor.Format) * sourceArray.Array3DDescriptor.NumChannels;

            object[] paramMemcpy3D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy3D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2), paramMemcpy3D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy3D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2(ref copyParams);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy array to array
        /// </summary>
        /// <param name="destArray"></param>
        public void CopyFromThisToArray(DispatchedCudaArray3D destArray)
        {
            CUDAMemCpy3D copyParams = new CUDAMemCpy3D();
            copyParams.srcArray = _cuArray;
            copyParams.srcMemoryType = CUMemoryType.Array;
            copyParams.dstArray = destArray.CUArray;
            copyParams.dstMemoryType = CUMemoryType.Array;
            copyParams.Depth = destArray._array3DDescriptor.Depth;
            copyParams.Height = destArray.Array3DDescriptor.Height;
            copyParams.WidthInBytes = destArray.Array3DDescriptor.Width * CudaHelperMethods.GetChannelSize(destArray.Array3DDescriptor.Format) * destArray.Array3DDescriptor.NumChannels;

            object[] paramMemcpy3D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy3D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2), paramMemcpy3D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy3D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy3D_v2(ref copyParams);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        #endregion

        #region Properties
        /// <summary>
        /// Returns the wrapped CUarray
        /// </summary>
        public CUarray CUArray
        {
            get { return _cuArray; }
        }

        /// <summary>
        /// Returns the wrapped CUDAArray3DDescriptor
        /// </summary>
        public CUDAArray3DDescriptor Array3DDescriptor
        {
            get { return _array3DDescriptor; }
        }

        /// <summary>
        /// Returns the Depth of the array
        /// </summary>
        public SizeT Depth
        {
            get { return _array3DDescriptor.Depth; }
        }

        /// <summary>
        /// Returns the Height of the array
        /// </summary>
        public SizeT Height
        {
            get { return _array3DDescriptor.Height; }
        }

        /// <summary>
        /// Returns the array width in elements
        /// </summary>
        public SizeT Width
        {
            get { return _array3DDescriptor.Width; }
        }

        /// <summary>
        /// Returns the array width in bytes
        /// </summary>
        public SizeT WidthInBytes
        {
            get { return _array3DDescriptor.Width * _array3DDescriptor.NumChannels * CudaHelperMethods.GetChannelSize(_array3DDescriptor.Format); }
        }
        #endregion
    }
}
