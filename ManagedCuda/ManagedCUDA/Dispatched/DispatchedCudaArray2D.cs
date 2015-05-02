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
    public enum CudaArray2DNumChannels
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
    public class DispatchedCudaArray2D : DispatchedCudaBaseClass, IDisposable
    {
        CUDAArrayDescriptor _arrayDescriptor;
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
        /// <param name="numChannels"></param>
        /// <param name="dispatcher"></param>
        public DispatchedCudaArray2D(CUArrayFormat format, SizeT height, SizeT width, CudaArray2DNumChannels numChannels, System.Windows.Threading.Dispatcher dispatcher)
            : base(dispatcher)
        {
            _arrayDescriptor = new CUDAArrayDescriptor();
            _arrayDescriptor.Format = format;
            _arrayDescriptor.Height = height;
            _arrayDescriptor.Width = width;
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
        public DispatchedCudaArray2D(CUarray cuArray, System.Windows.Threading.Dispatcher dispatcher)
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
        ~DispatchedCudaArray2D()
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
        /// <summary>
        /// A raw data copy method
        /// </summary>
        /// <param name="copyParameters">2D copy paramters</param>
        public void CopyData(CUDAMemCpy2D copyParameters)
        {
            object[] paramMemcpy2D = { copyParameters };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy2D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2), paramMemcpy2D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref aCopyParameters);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// A raw unaligned copy method
        /// </summary>
        /// <param name="copyParameters"></param>
        public void CopyDataUnaligned(CUDAMemCpy2D copyParameters)
        {
            object[] paramMemcpy2D = { copyParameters };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy2DUnaligned(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2DUnaligned_v2), paramMemcpy2D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2Unaligned(ref aCopyParameters);
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
            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.srcHost = hostSrc;
            copyParams.srcMemoryType = CUMemoryType.Host;
            copyParams.dstArray = _cuArray;
            copyParams.dstMemoryType = CUMemoryType.Array;
            copyParams.Height = _arrayDescriptor.Height;
            copyParams.WidthInBytes = _arrayDescriptor.Width * elementSizeInBytes;

            object[] paramMemcpy2D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy2D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2), paramMemcpy2D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
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
            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.dstHost = hostDest;
            copyParams.dstMemoryType = CUMemoryType.Host;
            copyParams.srcArray = _cuArray;
            copyParams.srcMemoryType = CUMemoryType.Array;
            copyParams.Height = _arrayDescriptor.Height;
            copyParams.WidthInBytes = _arrayDescriptor.Width * elementSizeInBytes;

            object[] paramMemcpy2D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy2D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2), paramMemcpy2D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy from Host to this array
        /// </summary>
        /// <typeparam name="T">Host array base type</typeparam>
        /// <param name="hostSrc">Source</param>
        public void CopyFromHostToThis<T>(T[] hostSrc) where T : struct
        {
            GCHandle handle = GCHandle.Alloc(hostSrc, GCHandleType.Pinned);

            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.srcHost = handle.AddrOfPinnedObject();
            copyParams.srcMemoryType = CUMemoryType.Host;
            copyParams.dstArray = _cuArray;
            copyParams.dstMemoryType = CUMemoryType.Array;
            copyParams.Height = _arrayDescriptor.Height;
            copyParams.WidthInBytes = _arrayDescriptor.Width * (uint)Marshal.SizeOf(typeof(T));

            object[] paramMemcpy2D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy2D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2), paramMemcpy2D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
            handle.Free();
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy data from this array to host
        /// </summary>
        /// <typeparam name="T">Host array base type</typeparam>
        /// <param name="hostDest">Destination</param>
        public void CopyFromThisToHost<T>(T[] hostDest) where T : struct
        {
            GCHandle handle = GCHandle.Alloc(hostDest, GCHandleType.Pinned);

            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.dstHost = handle.AddrOfPinnedObject();
            copyParams.dstMemoryType = CUMemoryType.Host;
            copyParams.srcArray = _cuArray;
            copyParams.srcMemoryType = CUMemoryType.Array;
            copyParams.Height = _arrayDescriptor.Height;
            copyParams.WidthInBytes = _arrayDescriptor.Width * (uint)Marshal.SizeOf(typeof(T));

            object[] paramMemcpy2D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy2D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2), paramMemcpy2D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
            handle.Free();
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy from a pitched device variable to this array
        /// </summary>
        /// <typeparam name="T">device variable base type</typeparam>
        /// <param name="deviceVariable">Source</param>
        public void CopyFromDeviceToThis<T>(DispatchedCudaPitchedDeviceVariable<T> deviceVariable) where T : struct
        {
            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.srcDevice = deviceVariable.DevicePointer;
            copyParams.srcMemoryType = CUMemoryType.Device;
            copyParams.srcPitch = deviceVariable.Pitch;
            copyParams.dstArray = _cuArray;
            copyParams.dstMemoryType = CUMemoryType.Array;
            copyParams.Height = deviceVariable.Height;
            copyParams.WidthInBytes = deviceVariable.WidthInBytes;

            object[] paramMemcpy2D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy2D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2), paramMemcpy2D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy from this array to a pitched device variable
        /// </summary>
        /// <typeparam name="T">device variable base type</typeparam>
        /// <param name="deviceVariable">Destination</param>
        public void CopyFromThisToDevice<T>(DispatchedCudaPitchedDeviceVariable<T> deviceVariable) where T : struct
        {
            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.dstDevice = deviceVariable.DevicePointer;
            copyParams.dstMemoryType = CUMemoryType.Device;
            copyParams.dstPitch = deviceVariable.Pitch;
            copyParams.srcArray = _cuArray;
            copyParams.srcMemoryType = CUMemoryType.Array;
            copyParams.Height = deviceVariable.Height;
            copyParams.WidthInBytes = deviceVariable.WidthInBytes;

            object[] paramMemcpy2D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy2D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2), paramMemcpy2D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy array to array
        /// </summary>
        /// <param name="sourceArray"></param>
        public void CopyFromArrayToThis(DispatchedCudaArray2D sourceArray)
        {
            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.srcArray = sourceArray.CUArray;
            copyParams.srcMemoryType = CUMemoryType.Array;
            copyParams.dstArray = _cuArray;
            copyParams.dstMemoryType = CUMemoryType.Array;
            copyParams.Height = sourceArray.ArrayDescriptor.Height;
            copyParams.WidthInBytes = sourceArray.ArrayDescriptor.Width * CudaHelperMethods.GetChannelSize(sourceArray.ArrayDescriptor.Format) * sourceArray.ArrayDescriptor.NumChannels;

            object[] paramMemcpy2D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy2D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2), paramMemcpy2D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Copy array to array
        /// </summary>
        /// <param name="destArray"></param>
        public void CopyFromThisToArray(DispatchedCudaArray2D destArray)
        {
            CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
            copyParams.srcArray = _cuArray;
            copyParams.srcMemoryType = CUMemoryType.Array;
            copyParams.dstArray = destArray.CUArray;
            copyParams.dstMemoryType = CUMemoryType.Array;
            copyParams.Height = destArray.ArrayDescriptor.Height;
            copyParams.WidthInBytes = destArray.ArrayDescriptor.Width * CudaHelperMethods.GetChannelSize(destArray.ArrayDescriptor.Format) * destArray.ArrayDescriptor.NumChannels;

            object[] paramMemcpy2D = { copyParams };

            res = (CUResult)_dispatcher.Invoke(new cuMemcpy2D(DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2), paramMemcpy2D);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
            //res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
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
        /// Returns the wrapped CUDAArrayDescriptor
        /// </summary>
        public CUDAArrayDescriptor ArrayDescriptor
        {
            get { return _arrayDescriptor; }
        }

        /// <summary>
        /// Returns the Height of the array
        /// </summary>
        public SizeT Height
        {
            get { return _arrayDescriptor.Height; }
        }

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
        #endregion
    }
}
