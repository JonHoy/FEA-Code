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
    /// CudaArrayTexture2D
    /// </summary>
    [Obsolete("CUDA 4.0 API calls are now thread safe. Dispatched... wrapper classes are of no use now.")]
    public class DispatchedCudaArrayTexture2D : DispatchedCudaBaseClass, IDisposable
    {
        CUtexref _texref;
        CUFilterMode _filtermode;
        CUTexRefSetFlags _flags;
        CUAddressMode _addressMode0, _addressMode1;
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
        /// Creates a new 2D texture from array memory. Allocates a new 2D array.
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="texName"></param>
        /// <param name="addressModeForAllDimensions"></param>
        /// <param name="filterMode"></param>
        /// <param name="flags"></param>
        /// <param name="format"></param>
        /// <param name="height">In elements</param>
        /// <param name="width">In elements</param>
        /// <param name="numChannels">1,2 or 4</param>
        public DispatchedCudaArrayTexture2D(DispatchedCudaKernel kernel, string texName, CUAddressMode addressModeForAllDimensions, CUFilterMode filterMode, CUTexRefSetFlags flags, CUArrayFormat format, SizeT height, SizeT width, CudaArray2DNumChannels numChannels)
            : this(kernel, texName, addressModeForAllDimensions, addressModeForAllDimensions, filterMode, flags, format, height, width, numChannels)
        {

        }

        /// <summary>
        /// Creates a new 2D texture from array memory. Allocates new array.
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="texName"></param>
        /// <param name="addressMode0"></param>
        /// <param name="addressMode1"></param>
        /// <param name="filterMode"></param>
        /// <param name="flags"></param>
        /// <param name="format"></param>
        /// <param name="height">In elements</param>
        /// <param name="width">In elements</param>
        /// <param name="numChannels">1,2 or 4</param>
        public DispatchedCudaArrayTexture2D(DispatchedCudaKernel kernel, string texName, CUAddressMode addressMode0, CUAddressMode addressMode1, CUFilterMode filterMode, CUTexRefSetFlags flags, CUArrayFormat format, SizeT height, SizeT width, CudaArray2DNumChannels numChannels)
            : base(kernel.Dispatcher)
        {
            _texref = new CUtexref();

            object[] paramModuleGetTexRef = { _texref, kernel.CUModule, texName };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetTexRef(DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef), paramModuleGetTexRef);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Texture name: {3}", DateTime.Now, "cuModuleGetTexRef", res, texName));
            //res = DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef(ref _texref, module, texName);
            if (res != CUResult.Success) throw new CudaException(res);
            _texref = (CUtexref)paramModuleGetTexRef[0];

            res = (CUResult)_dispatcher.Invoke(new cuTexRefSetAddressMode(DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode), _texref, 0, addressMode0);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
            //res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(_texref, 0, addressMode0);
            if (res != CUResult.Success) throw new CudaException(res);

            res = (CUResult)_dispatcher.Invoke(new cuTexRefSetAddressMode(DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode), _texref, 1, addressMode1);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
            //res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(_texref, 1, addressMode1);
            if (res != CUResult.Success) throw new CudaException(res);

            res = (CUResult)_dispatcher.Invoke(new cuTexRefSetFilterMode(DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode), _texref, filterMode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
            //res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(_texref, filterMode);
            if (res != CUResult.Success) throw new CudaException(res);

            res = (CUResult)_dispatcher.Invoke(new cuTexRefSetFlags(DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags), _texref, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
            //res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(_texref, flags);
            if (res != CUResult.Success) throw new CudaException(res);

            res = (CUResult)_dispatcher.Invoke(new cuTexRefSetFormat(DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat), _texref, format, (int)numChannels);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
            //res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(_texref, format, numChannels);
            if (res != CUResult.Success) throw new CudaException(res);

            _filtermode = filterMode;
            _flags = flags;
            _addressMode0 = addressMode0;
            _addressMode1 = addressMode1;
            _format = format;
            _height = height;
            _width = width;
            _numChannels = (int)numChannels;
            _name = texName;
            _module = kernel.CUModule;
            _cufunction = kernel.CUFunction;

            _channelSize = CudaHelperMethods.GetChannelSize(format);
            _dataSize = height * width * _numChannels * _channelSize;
            _array = new DispatchedCudaArray2D(format, height, width, (CudaArray2DNumChannels)numChannels, kernel.Dispatcher);


            res = (CUResult)_dispatcher.Invoke(new cuTexRefSetArray(DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetArray), _texref, _array.CUArray, CUTexRefSetArrayFlags.OverrideFormat);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetArray", res));
            //res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetArray(_texref, _array.CUArray, CUTexRefSetArrayFlags.OverrideFormat);
            if (res != CUResult.Success) throw new CudaException(res);

            //res = (CUResult)_dispatcher.Invoke(new cuParamSetTexRef(DriverAPINativeMethods.ParameterManagement.cuParamSetTexRef), kernel.CUFunction, CUParameterTexRef.Default, _texref);
            //Debug.WriteLine("{0:G}, {1}: {2}", DateTime.Now, "cuParamSetTexRef", res);
            ////res = DriverAPINativeMethods.ParameterManagement.cuParamSetTexRef(cufunction, CUParameterTexRef.Default, _texref);
            //if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Creates a new 2D texture from array memory. Allocates a new 2D array.
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="texName"></param>
        /// <param name="addressModeForAllDimensions"></param>
        /// <param name="filterMode"></param>
        /// <param name="flags"></param>
        /// <param name="array"></param>
        public DispatchedCudaArrayTexture2D(DispatchedCudaKernel kernel, string texName, CUAddressMode addressModeForAllDimensions, CUFilterMode filterMode, CUTexRefSetFlags flags, DispatchedCudaArray2D array)
            : this(kernel, texName, addressModeForAllDimensions, addressModeForAllDimensions, filterMode, flags, array)
        {

        }

        /// <summary>
        /// Creates a new 2D texture from array memory
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="texName"></param>
        /// <param name="addressMode0"></param>
        /// <param name="addressMode1"></param>
        /// <param name="filterMode"></param>
        /// <param name="flags"></param>
        /// <param name="array"></param>
        public DispatchedCudaArrayTexture2D(DispatchedCudaKernel kernel, string texName, CUAddressMode addressMode0, CUAddressMode addressMode1, CUFilterMode filterMode, CUTexRefSetFlags flags, DispatchedCudaArray2D array)
            : base(kernel.Dispatcher)
        {
            _texref = new CUtexref();

            object[] paramModuleGetTexRef = { _texref, kernel.CUModule, texName };
            res = (CUResult)_dispatcher.Invoke(new cuModuleGetTexRef(DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef), paramModuleGetTexRef);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Texture name: {3}", DateTime.Now, "cuModuleGetTexRef", res, texName));
            //res = DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef(ref _texref, module, texName);
            if (res != CUResult.Success) throw new CudaException(res);
            _texref = (CUtexref)paramModuleGetTexRef[0];

            res = (CUResult)_dispatcher.Invoke(new cuTexRefSetAddressMode(DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode), _texref, 0, addressMode0);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
            //res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(_texref, 0, addressMode0);
            if (res != CUResult.Success) throw new CudaException(res);

            res = (CUResult)_dispatcher.Invoke(new cuTexRefSetAddressMode(DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode), _texref, 1, addressMode1);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
            //res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(_texref, 1, addressMode1);
            if (res != CUResult.Success) throw new CudaException(res);

            res = (CUResult)_dispatcher.Invoke(new cuTexRefSetFilterMode(DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode), _texref, filterMode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
            //res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(_texref, filterMode);
            if (res != CUResult.Success) throw new CudaException(res);

            res = (CUResult)_dispatcher.Invoke(new cuTexRefSetFlags(DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags), _texref, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
            //res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(_texref, flags);
            if (res != CUResult.Success) throw new CudaException(res);

            res = (CUResult)_dispatcher.Invoke(new cuTexRefSetFormat(DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat), _texref, array.ArrayDescriptor.Format, (int)array.ArrayDescriptor.NumChannels);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
            //res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(_texref, format, numChannels);
            if (res != CUResult.Success) throw new CudaException(res);

            _filtermode = filterMode;
            _flags = flags;
            _addressMode0 = addressMode0;
            _addressMode1 = addressMode1;
            _format = array.ArrayDescriptor.Format;
            _height = array.Height;
            _width = array.Width;
            _numChannels = (int)array.ArrayDescriptor.NumChannels;
            _name = texName;
            _module = kernel.CUModule;
            _cufunction = kernel.CUFunction;

            _channelSize = CudaHelperMethods.GetChannelSize(array.ArrayDescriptor.Format);
            _dataSize = array.Height * array.Width * array.ArrayDescriptor.NumChannels * _channelSize;
            _array = array;

            res = (CUResult)_dispatcher.Invoke(new cuTexRefSetArray(DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetArray), _texref, _array.CUArray, CUTexRefSetArrayFlags.OverrideFormat);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetArray", res));
            //res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetArray(_texref, _array.CUArray, CUTexRefSetArrayFlags.OverrideFormat);
            if (res != CUResult.Success) throw new CudaException(res);

            //res = (CUResult)_dispatcher.Invoke(new cuParamSetTexRef(DriverAPINativeMethods.ParameterManagement.cuParamSetTexRef), kernel.CUFunction, CUParameterTexRef.Default, _texref);
            //Debug.WriteLine("{0:G}, {1}: {2}", DateTime.Now, "cuParamSetTexRef", res);
            ////res = DriverAPINativeMethods.ParameterManagement.cuParamSetTexRef(cufunction, CUParameterTexRef.Default, _texref);
            //if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~DispatchedCudaArrayTexture2D()
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
                // the _texref reference is not destroyed explicitly, as it is done automatically when module is unloaded
            }
        }
        #endregion

        #region Properties
        /// <summary>
        /// TextureReference
        /// </summary>
        public CUtexref TextureReference
        {
            get { return _texref; }
        }

        /// <summary>
        /// Flags
        /// </summary>
        public CUTexRefSetFlags Flags
        {
            get { return _flags; }
        }

        /// <summary>
        /// AddressMode0
        /// </summary>
        public CUAddressMode AddressMode0
        {
            get { return _addressMode0; }
        }

        /// <summary>
        /// AddressMode1
        /// </summary>
        public CUAddressMode AddressMode1
        {
            get { return _addressMode1; }
        }

        /// <summary>
        /// Format
        /// </summary>
        public CUArrayFormat Format
        {
            get { return _format; }
        }

        /// <summary>
        /// Filtermode
        /// </summary>
        public CUFilterMode Filtermode
        {
            get { return _filtermode; }
        }

        /// <summary>
        /// Width
        /// </summary>
        public SizeT Width
        {
            get { return _width; }
        }

        /// <summary>
        /// Height
        /// </summary>
        public SizeT Height
        {
            get { return _height; }
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
