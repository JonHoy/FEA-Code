using System;
using System.Collections.Generic;
using System.Text;
using CUDA.VectorTypes;

namespace CUDA.Dispatched
{
    /// <summary>
    /// CUDA device properties
    /// </summary>
    //public class DeviceProperties
    //{
    //    // Fields
    //    private int _clockRate;
    //    private dim3 _maxBlockDim;
    //    private dim3 _maxGridDim;
    //    private int _maxThreadsPerBlock;
    //    private int _memPitch;
    //    private int _regsPerBlock;
    //    private int _sharedMemPerBlock;
    //    private int _textureAlign;
    //    private int _totalConstantMemory;
    //    private string _deviceName;
    //    private Version _computeCapability;
    //    private Version _driverVersion;
    //    private uint _totalGlobalMemory;
    //    private int _multiProcessorCount;
    //    private int _warpSize;
    //    private bool _gpuOverlap;
    //    private bool _kernelExecTimeoutEnabled;
    //    private bool _integrated;
    //    private bool _canMapHostMemory;
    //    private BasicTypes.CUComputeMode _computeMode;
    //    private int _maximumTexture1DWidth;
    //    private int _maximumTexture2DWidth;
    //    private int _maximumTexture2DHeight;
    //    private int _maximumTexture3DWidth;
    //    private int _maximumTexture3DHeight;
    //    private int _maximumTexture3DDepth;
    //    private int _maximumTexture2DArray_Width;
    //    private int _maximumTexture2DArray_Height;
    //    private int _maximumTexture2DArray_NumSlices;
    //    private int _surfaceAllignment;
    //    private bool _concurrentKernels;
    //    private bool _ECCEnabled;
    //    private int _PCIBusID;
    //    private int _PCIDeviceID;


    //    // Properties
    //    /// <summary>
    //    /// Peak clock frequency in kilohertz
    //    /// </summary>
    //    public int ClockRate
    //    {
    //        get
    //        {
    //            return this._clockRate;
    //        }
    //        internal set
    //        {
    //            this._clockRate = value;
    //        }
    //    }

    //    /// <summary>
    //    /// Maximum block dimensions
    //    /// </summary>
    //    public dim3 MaxBlockDim
    //    {
    //        get
    //        {
    //            return this._maxBlockDim;
    //        }
    //        internal set
    //        {
    //            this._maxBlockDim = value;
    //        }
    //    }

    //    /// <summary>
    //    /// Maximum grid dimensions
    //    /// </summary>
    //    public dim3 MaxGridDim
    //    {
    //        get
    //        {
    //            return this._maxGridDim;
    //        }
    //        internal set
    //        {
    //            this._maxGridDim = value;
    //        }
    //    }

    //    /// <summary>
    //    /// Maximum number of threads per block
    //    /// </summary>
    //    public int MaxThreadsPerBlock
    //    {
    //        get
    //        {
    //            return this._maxThreadsPerBlock;
    //        }
    //        internal set
    //        {
    //            this._maxThreadsPerBlock = value;
    //        }
    //    }

    //    /// <summary>
    //    /// Maximum pitch in bytes allowed by memory copies
    //    /// </summary>
    //    public int MemoryPitch
    //    {
    //        get
    //        {
    //            return this._memPitch;
    //        }
    //        internal set
    //        {
    //            this._memPitch = value;
    //        }
    //    }

    //    /// <summary>
    //    /// Maximum number of 32-bit registers available per block
    //    /// </summary>
    //    public int RegistersPerBlock
    //    {
    //        get
    //        {
    //            return this._regsPerBlock;
    //        }
    //        internal set
    //        {
    //            this._regsPerBlock = value;
    //        }
    //    }

    //    /// <summary>
    //    /// Maximum shared memory available per block in bytes
    //    /// </summary>
    //    public int SharedMemoryPerBlock
    //    {
    //        get
    //        {
    //            return this._sharedMemPerBlock;
    //        }
    //        internal set
    //        {
    //            this._sharedMemPerBlock = value;
    //        }
    //    }

    //    /// <summary>
    //    /// Alignment requirement for textures
    //    /// </summary>
    //    public int TextureAlign
    //    {
    //        get
    //        {
    //            return this._textureAlign;
    //        }
    //        set
    //        {
    //            this._textureAlign = value;
    //        }
    //    }

    //    /// <summary>
    //    /// Memory available on device for __constant__ variables in a CUDA C kernel in bytes
    //    /// </summary>
    //    public int TotalConstantMemory
    //    {
    //        get
    //        {
    //            return this._totalConstantMemory;
    //        }
    //        internal set
    //        {
    //            this._totalConstantMemory = value;
    //        }
    //    }

    //    /// <summary>
    //    /// Name of the device
    //    /// </summary>
    //    public string DeviceName
    //    {
    //        get { return this._deviceName; }
    //        internal set { this._deviceName = value; }
    //    }

    //    /// <summary>
    //    /// Compute capability version
    //    /// </summary>
    //    public Version ComputeCapability
    //    {
    //        get { return this._computeCapability; }
    //        internal set { this._computeCapability = value; }
    //    }

    //    /// <summary>
    //    /// Driver version
    //    /// </summary>
    //    public Version DriverVersion
    //    {
    //        get { return this._driverVersion; }
    //        internal set { this._driverVersion = value; }
    //    }

    //    /// <summary>
    //    /// Total amount of global memory on the device
    //    /// </summary>
    //    public uint TotalGlobalMemory
    //    {
    //        get { return this._totalGlobalMemory; }
    //        internal set { this._totalGlobalMemory = value; }
    //    }

    //    /// <summary>
    //    /// Number of multiprocessors on device
    //    /// </summary>
    //    public int MultiProcessorCount
    //    {
    //        get { return this._multiProcessorCount; }
    //        internal set { this._multiProcessorCount = value; }
    //    }

    //    /// <summary>
    //    /// Warp size in threads (also called SIMDWith)
    //    /// </summary>
    //    public int WarpSize
    //    {
    //        get { return this._warpSize; }
    //        internal set { this._warpSize = value; }
    //    }

    //    /// <summary>
    //    /// Device can possibly copy memory and execute a kernel concurrently
    //    /// </summary>
    //    public bool GpuOverlap
    //    {
    //        get { return this._gpuOverlap; }
    //        internal set { this._gpuOverlap = value; }
    //    }

    //    /// <summary>
    //    /// Specifies whether there is a run time limit on kernels
    //    /// </summary>
    //    public bool KernelExecTimeoutEnabled
    //    {
    //        get { return this._kernelExecTimeoutEnabled; }
    //        internal set { this._kernelExecTimeoutEnabled = value; }
    //    }

    //    /// <summary>
    //    /// Device is integrated with host memory
    //    /// </summary>
    //    public bool Integrated
    //    {
    //        get { return this._integrated; }
    //        internal set { this._integrated = value; }
    //    }

    //    /// <summary>
    //    /// Device can map host memory into CUDA address space
    //    /// </summary>
    //    public bool CanMapHostMemory
    //    {
    //        get { return this._canMapHostMemory; }
    //        internal set { this._canMapHostMemory = value; }
    //    }

    //    /// <summary>
    //    /// Compute mode (See CUComputeMode for details)
    //    /// </summary>
    //    public BasicTypes.CUComputeMode ComputeMode
    //    {
    //        get { return this._computeMode; }
    //        internal set { this._computeMode = value; }
    //    }


    //    /// <summary>
    //    /// Maximum 1D texture width
    //    /// </summary>
    //    public int MaximumTexture1DWidth
    //    {
    //        get { return this._maximumTexture1DWidth; }
    //        internal set { this._maximumTexture1DWidth = value; }
    //    }

    //    /// <summary>
    //    /// Maximum 2D texture width
    //    /// </summary>
    //    public int MaximumTexture2DWidth
    //    {
    //        get { return this._maximumTexture2DWidth; }
    //        internal set { this._maximumTexture2DWidth = value; }
    //    }

    //    /// <summary>
    //    /// Maximum 2D texture height
    //    /// </summary>
    //    public int MaximumTexture2DHeight
    //    {
    //        get { return this._maximumTexture2DHeight; }
    //        internal set { this._maximumTexture2DHeight = value; }
    //    }

    //    /// <summary>
    //    /// Maximum 3D texture width
    //    /// </summary>
    //    public int MaximumTexture3DWidth
    //    {
    //        get { return this._maximumTexture3DWidth; }
    //        internal set { this._maximumTexture3DWidth = value; }
    //    }

    //    /// <summary>
    //    /// Maximum 3D texture height
    //    /// </summary>
    //    public int MaximumTexture3DHeight
    //    {
    //        get { return this._maximumTexture3DHeight; }
    //        internal set { this._maximumTexture3DHeight = value; }
    //    }

    //    /// <summary>
    //    /// Maximum 3D texture depth
    //    /// </summary>
    //    public int MaximumTexture3DDepth
    //    {
    //        get { return this._maximumTexture3DDepth; }
    //        internal set { this._maximumTexture3DDepth = value; }
    //    }

    //    /// <summary>
    //    /// Maximum texture array width
    //    /// </summary>
    //    public int MaximumTexture2DArray_Width
    //    {
    //        get { return this._maximumTexture2DArray_Width; }
    //        internal set { this._maximumTexture2DArray_Width = value; }
    //    }

    //    /// <summary>
    //    /// Maximum texture array height
    //    /// </summary>
    //    public int MaximumTexture2DArray_Height
    //    {
    //        get { return this._maximumTexture2DArray_Height; }
    //        internal set { this._maximumTexture2DArray_Height = value; }
    //    }

    //    /// <summary>
    //    /// Maximum slices in a texture array
    //    /// </summary>
    //    public int MaximumTexture2DArray_NumSlices
    //    {
    //        get { return this._maximumTexture2DArray_NumSlices; }
    //        internal set { this._maximumTexture2DArray_NumSlices = value; }
    //    }

    //    /// <summary>
    //    /// Alignment requirement for surfaces
    //    /// </summary>
    //    public int SurfaceAllignment
    //    {
    //        get { return this._surfaceAllignment; }
    //        internal set { this._surfaceAllignment = value; }
    //    }

    //    /// <summary>
    //    /// Device can possibly execute multiple kernels concurrently
    //    /// </summary>
    //    public bool ConcurrentKernels
    //    {
    //        get { return this._concurrentKernels; }
    //        internal set { this._concurrentKernels = value; }
    //    }

    //    /// <summary>
    //    /// Device has ECC support enabled
    //    /// </summary>
    //    public bool ECCEnabled
    //    {
    //        get { return this._ECCEnabled; }
    //        internal set { this._ECCEnabled = value; }
    //    }

    //    /// <summary>
    //    /// PCI bus ID of the device
    //    /// </summary>
    //    public int PCIBusID
    //    {
    //        get { return this._PCIBusID; }
    //        internal set { this._PCIBusID = value; }
    //    }

    //    /// <summary>
    //    /// PCI device ID of the device
    //    /// </summary>
    //    public int PCIDeviceID
    //    {
    //        get { return this._PCIDeviceID; }
    //        internal set { this._PCIDeviceID = value; }
    //    }
    //}
}
