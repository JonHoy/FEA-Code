using System;
using System.Collections.Generic;
using System.Text;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace ManagedCuda.Dispatched
{
    /// <summary>
    /// CudaSurface2D
    /// </summary>
    [Obsolete("CUDA 4.0 API calls are now thread safe. Dispatched... wrapper classes are of no use now.")]
    public class DispatchedCudaSurface2D : DispatchedCudaBaseClass, IDisposable
    {
        CUsurfref _surfref;
        CUSurfRefSetFlags _flags;
        CUArrayFormat _format;
        SizeT _height;
        SizeT _width;
        uint _channelSize;
        SizeT _dataSize;
        int _numChannels;
        string _name;
        CUmodule _module;
        CUfunction _cufunction;
        DispatchedCudaArray2D _array;
        CUResult res;
        bool disposed;

        #region Constructors
        /// <summary>
        /// Creates a new 1D surface from array memory. Allocates new array.
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="surfName"></param>
        /// <param name="flags"></param>
        /// <param name="format"></param>
        /// <param name="height">In elements</param>
        /// <param name="width">In elements</param>
        /// <param name="numChannels"></param>
        public DispatchedCudaSurface2D(DispatchedCudaKernel kernel, string surfName, CUSurfRefSetFlags flags, CUArrayFormat format, SizeT height, SizeT width, CudaArray2DNumChannels numChannels)
            : base(kernel.Dispatcher)
        {
            _surfref = new CUsurfref();
            object[] paramModuleGetSurfRef = { _surfref, kernel.CUModule, surfName };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetSurfRef(DriverAPINativeMethods.ModuleManagement.cuModuleGetSurfRef), paramModuleGetSurfRef);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Surface name: {3}", DateTime.Now, "cuModuleGetSurfRef", res, surfName));
            if (res != CUResult.Success) throw new CudaException(res);
            _surfref = (CUsurfref)paramModuleGetSurfRef[0];

            _flags = flags;
            _format = format;
            _height = height;
            _width = width;
            _numChannels = (int)numChannels;
            _name = surfName;
            _module = kernel.CUModule;
            _cufunction = kernel.CUFunction;

            _channelSize = CudaHelperMethods.GetChannelSize(format);
            _dataSize = height * width * _numChannels * _channelSize;
            _array = new DispatchedCudaArray2D(format, height, width, numChannels, kernel.Dispatcher);

            res = (CUResult)_dispatcher.Invoke(new cuSurfRefSetArray(DriverAPINativeMethods.SurfaceReferenceManagement.cuSurfRefSetArray), _surfref, _array.CUArray, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuSurfRefSetArray", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Creates a new 1D surface from array memory.
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="surfName"></param>
        /// <param name="flags"></param>
        /// <param name="array"></param>
        public DispatchedCudaSurface2D(DispatchedCudaKernel kernel, string surfName, CUSurfRefSetFlags flags, DispatchedCudaArray2D array)
            : base(kernel.Dispatcher)
        {
            _surfref = new CUsurfref();
            object[] paramModuleGetSurfRef = { _surfref, kernel.CUModule, surfName };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetSurfRef(DriverAPINativeMethods.ModuleManagement.cuModuleGetSurfRef), paramModuleGetSurfRef);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Surface name: {3}", DateTime.Now, "cuModuleGetSurfRef", res, surfName));
            if (res != CUResult.Success) throw new CudaException(res);
            _surfref = (CUsurfref)paramModuleGetSurfRef[0];

            _flags = flags;
            _format = array.ArrayDescriptor.Format;
            _height = array.Height;
            _width = array.Width;
            _numChannels = (int)array.ArrayDescriptor.NumChannels;
            _name = surfName;
            _module = kernel.CUModule;
            _cufunction = kernel.CUFunction;
            _channelSize = CudaHelperMethods.GetChannelSize(array.ArrayDescriptor.Format);
            _dataSize = array.Height * array.Width * array.ArrayDescriptor.NumChannels * _channelSize;
            _array = array;

            res = (CUResult)_dispatcher.Invoke(new cuSurfRefSetArray(DriverAPINativeMethods.SurfaceReferenceManagement.cuSurfRefSetArray), _surfref, _array.CUArray, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuSurfRefSetArray", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~DispatchedCudaSurface2D()
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
                _array.Dispose();
                disposed = true;
                // the _surfref reference is not destroyed explicitly, as it is done automatically when module is unloaded
            }
        }
        #endregion

        #region Properties
        /// <summary>
        /// SurfaceReference
        /// </summary>
        public CUsurfref SurfaceReference
        {
            get { return _surfref; }
        }

        /// <summary>
        /// Flags
        /// </summary>
        public CUSurfRefSetFlags Flags
        {
            get { return _flags; }
        }

        /// <summary>
        /// Format
        /// </summary>
        public CUArrayFormat Format
        {
            get { return _format; }
        }

        /// <summary>
        /// Height
        /// </summary>
        public SizeT Height
        {
            get { return _height; }
        }

        /// <summary>
        /// Width
        /// </summary>
        public SizeT Width
        {
            get { return _width; }
        }

        /// <summary>
        /// ChannelSize
        /// </summary>
        public uint ChannelSize
        {
            get { return _channelSize; }
        }

        /// <summary>
        /// TotalSizeInBytes
        /// </summary>
        public SizeT TotalSizeInBytes
        {
            get { return _dataSize; }
        }

        /// <summary>
        /// NumChannels
        /// </summary>
        public int NumChannels
        {
            get { return _numChannels; }
        }

        /// <summary>
        /// Name
        /// </summary>
        public string Name
        {
            get { return _name; }
        }

        /// <summary>
        /// Module
        /// </summary>
        public CUmodule Module
        {
            get { return _module; }
        }

        /// <summary>
        /// CUFuntion
        /// </summary>
        public CUfunction CUFuntion
        {
            get { return _cufunction; }
        }

        /// <summary>
        /// Array
        /// </summary>
        public DispatchedCudaArray2D Array
        {
            get { return _array; }
        }
        #endregion
    }
}
