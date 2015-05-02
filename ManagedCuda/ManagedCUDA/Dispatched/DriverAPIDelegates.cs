using System;
using System.Collections.Generic;
using System.Text;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;


namespace ManagedCuda.Dispatched
{
    #region Device management
    
        delegate CUResult cuDeviceGet(ref CUdevice device, int ordinal);

        delegate CUResult cuDeviceGetCount(ref int count);

        delegate CUResult cuDeviceGetName(byte[] name, int len, CUdevice dev);

        delegate CUResult cuDeviceComputeCapability(ref int major, ref int minor, CUdevice dev);

        delegate CUResult cuDeviceTotalMem_v2(ref SizeT bytes, CUdevice dev);

        delegate CUResult cuDeviceGetProperties(ref CUDeviceProperties prop, CUdevice dev);

        delegate CUResult cuDeviceGetAttribute(ref int pi, CUDeviceAttribute attrib, CUdevice dev);
    
    #endregion

    #region Context management
    delegate CUResult cuCtxCreate_v2(ref CUcontext pctx, CUCtxFlags flags, CUdevice dev);

    delegate CUResult cuCtxDestroy(CUcontext ctx);

    delegate CUResult cuCtxAttach(ref CUcontext pctx, CUCtxAttachFlags flags);

    delegate CUResult cuCtxDetach(CUcontext ctx);

    delegate CUResult cuCtxPushCurrent(CUcontext ctx);

    delegate CUResult cuCtxPopCurrent(ref  CUcontext pctx);

    delegate CUResult cuCtxGetDevice(ref CUdevice device);

    delegate CUResult cuCtxSynchronize();

    delegate CUResult cuCtxGetApiVersion(CUcontext context, ref uint version);
    
    #endregion

    #region Module management
    delegate CUResult cuModuleLoad(ref CUmodule module, string fname);

    delegate CUResult cuModuleLoadData(ref CUmodule module,  byte[] image);

    delegate CUResult cuModuleLoadDataEx(ref CUmodule module,  byte[] image, uint numOptions,  CUJITOption[] options,  IntPtr[] optionValues);

    delegate CUResult cuModuleLoadFatBinary(ref CUmodule module,  byte[] fatCubin);

    delegate CUResult cuModuleUnload(CUmodule hmod);

    delegate CUResult cuModuleGetFunction(ref CUfunction hfunc, CUmodule hmod, string name);

    delegate CUResult cuModuleGetGlobal_v2(ref CUdeviceptr dptr, ref SizeT bytes, CUmodule hmod, string name);

    delegate CUResult cuModuleGetTexRef(ref CUtexref pTexRef, CUmodule hmod, string name);

    delegate CUResult cuModuleGetSurfRef(ref CUsurfref pSurfRef, CUmodule hmod, string name);
    
    #endregion

    #region Memory management
    delegate CUResult cuMemGetInfo_v2(ref SizeT free, ref SizeT total);

    delegate CUResult cuMemAlloc_v2(ref CUdeviceptr dptr, SizeT bytesize);

    delegate CUResult cuMemAllocPitch_v2(ref CUdeviceptr dptr, ref SizeT pPitch, SizeT WidthInBytes, SizeT Height, uint ElementSizeBytes);

    delegate CUResult cuMemFree_v2(CUdeviceptr dptr);

    delegate CUResult cuMemGetAddressRange_v2(ref CUdeviceptr pbase, ref SizeT psize, CUdeviceptr dptr);

    delegate CUResult cuMemAllocHost_v2(ref IntPtr pp, SizeT bytesize);

    delegate CUResult cuMemFreeHost(IntPtr p);

    delegate CUResult cuMemHostAlloc(ref IntPtr pp, SizeT bytesize, CUMemHostAllocFlags Flags);

    delegate CUResult cuMemHostGetDevicePointer_v2(ref CUdeviceptr pdptr, IntPtr p, int Flags);

    delegate CUResult cuMemHostGetFlags(ref CUMemHostAllocFlags pFlags, ref IntPtr p);
    
    #endregion

    #region Synchronous Memcpy
    #region VectorTypesArray
    delegate CUResult cuMemcpyHtoDDim3A(CUdeviceptr dstDevice, dim3[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDChar1A(CUdeviceptr dstDevice, char1[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDChar2A(CUdeviceptr dstDevice, char2[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDChar3A(CUdeviceptr dstDevice, char3[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDChar4A(CUdeviceptr dstDevice, char4[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUChar1A(CUdeviceptr dstDevice, uchar1[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUChar2A(CUdeviceptr dstDevice, uchar2[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUChar3A(CUdeviceptr dstDevice, uchar3[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUChar4A(CUdeviceptr dstDevice, uchar4[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDShort1A(CUdeviceptr dstDevice, short1[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDShort2A(CUdeviceptr dstDevice, short2[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDShort3A(CUdeviceptr dstDevice, short3[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDShort4A(CUdeviceptr dstDevice, short4[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUShort1A(CUdeviceptr dstDevice, ushort1[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUShort2A(CUdeviceptr dstDevice, ushort2[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUShort3A(CUdeviceptr dstDevice, ushort3[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUShort4A(CUdeviceptr dstDevice, ushort4[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDInt1A(CUdeviceptr dstDevice, int1[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDInt2A(CUdeviceptr dstDevice, int2[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDInt3A(CUdeviceptr dstDevice, int3[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDInt4A(CUdeviceptr dstDevice, int4[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUInt1A(CUdeviceptr dstDevice, uint1[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUInt2A(CUdeviceptr dstDevice, uint2[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUInt3A(CUdeviceptr dstDevice, uint3[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUInt4A(CUdeviceptr dstDevice, uint4[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDLong1A(CUdeviceptr dstDevice, long1[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDLong2A(CUdeviceptr dstDevice, long2[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDLong3A(CUdeviceptr dstDevice, long3[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDLong4A(CUdeviceptr dstDevice, long4[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDULong1A(CUdeviceptr dstDevice, ulong1[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDULong2A(CUdeviceptr dstDevice, ulong2[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDULong3A(CUdeviceptr dstDevice, ulong3[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDULong4A(CUdeviceptr dstDevice, ulong4[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDFloat1A(CUdeviceptr dstDevice, float1[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDFloat2A(CUdeviceptr dstDevice, float2[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDFloat3A(CUdeviceptr dstDevice, float3[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDFloat4A(CUdeviceptr dstDevice, float4[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDDouble1A(CUdeviceptr dstDevice, double1[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDDouble2A(CUdeviceptr dstDevice, double2[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDDoubleComplexA(CUdeviceptr dstDevice, cuDoubleComplex[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDDoubleRealA(CUdeviceptr dstDevice, cuDoubleReal[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDFloatComplexA(CUdeviceptr dstDevice, cuFloatComplex[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDFloatRealA(CUdeviceptr dstDevice, cuFloatReal[] srcHost, SizeT ByteCount);
    #endregion
    #region NumberTypesArray
    delegate CUResult cuMemcpyHtoDByteA(CUdeviceptr dstDevice, byte[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDSByteA(CUdeviceptr dstDevice, sbyte[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUShortA(CUdeviceptr dstDevice, ushort[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDShortA(CUdeviceptr dstDevice, short[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUIntA(CUdeviceptr dstDevice, uint[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDIntA(CUdeviceptr dstDevice, int[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDULongA(CUdeviceptr dstDevice, ulong[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDLongA(CUdeviceptr dstDevice, long[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDFloatA(CUdeviceptr dstDevice, float[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDDoubleA(CUdeviceptr dstDevice, double[] srcHost, SizeT ByteCount);
    #endregion
    #region VectorTypes
    delegate CUResult cuMemcpyHtoDDim3(CUdeviceptr dstDevice, ref dim3 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDChar1(CUdeviceptr dstDevice, ref char1 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDChar2(CUdeviceptr dstDevice, ref char2 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDChar3(CUdeviceptr dstDevice, ref char3 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDChar4(CUdeviceptr dstDevice, ref char4 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUChar1(CUdeviceptr dstDevice, ref uchar1 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUChar2(CUdeviceptr dstDevice, ref uchar2 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUChar3(CUdeviceptr dstDevice, ref uchar3 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUChar4(CUdeviceptr dstDevice, ref uchar4 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDShort1(CUdeviceptr dstDevice, ref short1 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDShort2(CUdeviceptr dstDevice, ref short2 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDShort3(CUdeviceptr dstDevice, ref short3 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDShort4(CUdeviceptr dstDevice, ref short4 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUShort1(CUdeviceptr dstDevice, ref ushort1 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUShort2(CUdeviceptr dstDevice, ref ushort2 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUShort3(CUdeviceptr dstDevice, ref ushort3 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUShort4(CUdeviceptr dstDevice, ref ushort4 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDInt1(CUdeviceptr dstDevice, ref int1 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDInt2(CUdeviceptr dstDevice, ref int2 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDInt3(CUdeviceptr dstDevice, ref int3 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDInt4(CUdeviceptr dstDevice, ref int4 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUInt1(CUdeviceptr dstDevice, ref uint1 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUInt2(CUdeviceptr dstDevice, ref uint2 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUInt3(CUdeviceptr dstDevice, ref uint3 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUInt4(CUdeviceptr dstDevice, ref uint4 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDLong1(CUdeviceptr dstDevice, ref long1 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDLong2(CUdeviceptr dstDevice, ref long2 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDLong3(CUdeviceptr dstDevice, ref long3 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDLong4(CUdeviceptr dstDevice, ref long4 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDULong1(CUdeviceptr dstDevice, ref ulong1 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDULong2(CUdeviceptr dstDevice, ref ulong2 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDULong3(CUdeviceptr dstDevice, ref ulong3 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDULong4(CUdeviceptr dstDevice, ref ulong4 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDFloat1(CUdeviceptr dstDevice, ref float1 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDFloat2(CUdeviceptr dstDevice, ref float2 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDFloat3(CUdeviceptr dstDevice, ref float3 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDFloat4(CUdeviceptr dstDevice, ref float4 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDDouble1(CUdeviceptr dstDevice, ref double1 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDDouble2(CUdeviceptr dstDevice, ref double2 srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDDoubleComplex(CUdeviceptr dstDevice, ref cuDoubleComplex srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDDoubleReal(CUdeviceptr dstDevice, ref cuDoubleReal srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDFloatComplex(CUdeviceptr dstDevice, ref cuFloatComplex srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDFloatReal(CUdeviceptr dstDevice, ref cuFloatReal srcHost, SizeT ByteCount);
    #endregion
    #region NumberTypes
    delegate CUResult cuMemcpyHtoDByte(CUdeviceptr dstDevice, ref byte srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDSByte(CUdeviceptr dstDevice, ref sbyte srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUShort(CUdeviceptr dstDevice, ref ushort srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDShort(CUdeviceptr dstDevice, ref short srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDUInt(CUdeviceptr dstDevice, ref uint srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDInt(CUdeviceptr dstDevice, ref int srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDULong(CUdeviceptr dstDevice, ref ulong srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDLong(CUdeviceptr dstDevice, ref long srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDFloat(CUdeviceptr dstDevice, ref float srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoDDouble(CUdeviceptr dstDevice, ref double srcHost, SizeT ByteCount);
    #endregion
    delegate CUResult cuMemcpyHtoDIntPtr(CUdeviceptr dstDevice, IntPtr srcHost, SizeT ByteCount);


    //Device to Host
    #region VectorTypesArray
    delegate CUResult cuMemcpyDtoHDim3A(dim3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHChar1A(char1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHChar2A(char2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHChar3A(char3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHChar4A(char4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUChar1A(uchar1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUChar2A(uchar2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUChar3A(uchar3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUChar4A(uchar4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHShort1A(short1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHShort2A(short2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHShort3A(short3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHShort4A(short4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUShort1A(ushort1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUShort2A(ushort2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUShort3A(ushort3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUShort4A(ushort4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHInt1A(int1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHInt2A(int2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHInt3A(int3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHInt4A(int4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUInt1A(uint1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUInt2A(uint2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUInt3A(uint3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUInt4A(uint4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHLong1A(long1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHLong2A(long2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHLong3A(long3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHLong4A(long4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHULong1A(ulong1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHULong2A(ulong2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHULong3A(ulong3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHULong4A(ulong4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHFloat1A(float1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHFloat2A(float2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHFloat3A(float3[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHFloat4A(float4[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHDouble1A(double1[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHDouble2A(double2[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHDoubleComplexA(cuDoubleComplex[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHDoubleRealA(cuDoubleReal[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHFloatComplexA(cuFloatComplex[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHFloatRealA(cuFloatReal[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    #endregion
    #region NumberTypesArray
    delegate CUResult cuMemcpyDtoHByteA(byte[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHSByteA(sbyte[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUShortA(ushort[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHShortA(short[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUIntA(uint[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHIntA(int[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHULongA(ulong[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHLongA(long[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHFloatA(float[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHDoubleA(double[] dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    #endregion
    #region VectorTypes
    delegate CUResult cuMemcpyDtoHDim3(ref dim3 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHChar1(ref char1 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHChar2(ref char2 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHChar3(ref char3 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHChar4(ref char4 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUChar1(ref uchar1 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUChar2(ref uchar2 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUChar3(ref uchar3 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUChar4(ref uchar4 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHShort1(ref short1 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHShort2(ref short2 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHShort3(ref short3 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHShort4(ref short4 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUShort1(ref ushort1 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUShort2(ref ushort2 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUShort3(ref ushort3 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUShort4(ref ushort4 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHInt1(ref int1 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHInt2(ref int2 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHInt3(ref int3 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHInt4(ref int4 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUInt1(ref uint1 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUInt2(ref uint2 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUInt3(ref uint3 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUInt4(ref uint4 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHLong1(ref long1 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHLong2(ref long2 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHLong3(ref long3 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHLong4(ref long4 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHULong1(ref ulong1 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHULong2(ref ulong2 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHULong3(ref ulong3 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHULong4(ref ulong4 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHFloat1(ref float1 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHFloat2(ref float2 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHFloat3(ref float3 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHFloat4(ref float4 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHDouble1(ref double1 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHDouble2(ref double2 dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHDoubleComplex(ref cuDoubleComplex dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHDoubleReal(ref cuDoubleReal dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHFloatComplex(ref cuFloatComplex dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHFloatReal(ref cuFloatReal dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    #endregion
    #region NumberTypes
    delegate CUResult cuMemcpyDtoHByte(ref byte dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHSByte(ref sbyte dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUShort(ref ushort dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHShort(ref short dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHUInt(ref uint dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHInt(ref int dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHULong(ref ulong dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHLong(ref long dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHFloat(ref float dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyDtoHDouble(ref double dstHost, CUdeviceptr srcDevice, SizeT ByteCount);
    #endregion
    delegate CUResult cuMemcpyDtoHIntPtr( IntPtr dstHost, CUdeviceptr srcDevice, SizeT ByteCount);

    // device <-> device memory
    delegate CUResult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, SizeT ByteCount);

    // device <-> array memory
    delegate CUResult cuMemcpyDtoA(CUarray dstArray, SizeT dstOffset, CUdeviceptr srcDevice, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);

    // system <-> array memory
    #region VectorTypesArray
    delegate CUResult cuMemcpyHtoADim3A(CUarray dstArray, SizeT dstOffset, dim3[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAChar1A(CUarray dstArray, SizeT dstOffset, char1[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAChar2A(CUarray dstArray, SizeT dstOffset, char2[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAChar3A(CUarray dstArray, SizeT dstOffset, char3[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAChar4A(CUarray dstArray, SizeT dstOffset, char4[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAUChar1A(CUarray dstArray, SizeT dstOffset, uchar1[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAUChar2A(CUarray dstArray, SizeT dstOffset, uchar2[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAUChar3A(CUarray dstArray, SizeT dstOffset, uchar3[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAUChar4A(CUarray dstArray, SizeT dstOffset, uchar4[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAShort1A(CUarray dstArray, SizeT dstOffset, short1[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAShort2A(CUarray dstArray, SizeT dstOffset, short2[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAShort3A(CUarray dstArray, SizeT dstOffset, short3[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAShort4A(CUarray dstArray, SizeT dstOffset, short4[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAUShort1A(CUarray dstArray, SizeT dstOffset, ushort1[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAUShort2A(CUarray dstArray, SizeT dstOffset, ushort2[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAUShort3A(CUarray dstArray, SizeT dstOffset, ushort3[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAUShort4A(CUarray dstArray, SizeT dstOffset, ushort4[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAInt1A(CUarray dstArray, SizeT dstOffset, int1[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAInt2A(CUarray dstArray, SizeT dstOffset, int2[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAInt3A(CUarray dstArray, SizeT dstOffset, int3[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAInt4A(CUarray dstArray, SizeT dstOffset, int4[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAUInt1A(CUarray dstArray, SizeT dstOffset, uint1[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAUInt2A(CUarray dstArray, SizeT dstOffset, uint2[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAUInt3A(CUarray dstArray, SizeT dstOffset, uint3[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAUInt4A(CUarray dstArray, SizeT dstOffset, uint4[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoALong1A(CUarray dstArray, SizeT dstOffset, long1[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoALong2A(CUarray dstArray, SizeT dstOffset, long2[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoALong3A(CUarray dstArray, SizeT dstOffset, long3[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoALong4A(CUarray dstArray, SizeT dstOffset, long4[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAULong1A(CUarray dstArray, SizeT dstOffset, ulong1[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAULong2A(CUarray dstArray, SizeT dstOffset, ulong2[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAULong3A(CUarray dstArray, SizeT dstOffset, ulong3[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAULong4A(CUarray dstArray, SizeT dstOffset, ulong4[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAFloat1A(CUarray dstArray, SizeT dstOffset, float1[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAFloat2A(CUarray dstArray, SizeT dstOffset, float2[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAFloat3A(CUarray dstArray, SizeT dstOffset, float3[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAFloat4A(CUarray dstArray, SizeT dstOffset, float4[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoADouble1A(CUarray dstArray, SizeT dstOffset, double1[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoADouble2A(CUarray dstArray, SizeT dstOffset, double2[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoADoubleComplexA(CUarray dstArray, SizeT dstOffset, cuDoubleComplex[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoADoubleRealA(CUarray dstArray, SizeT dstOffset, cuDoubleReal[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAFloatComplexA(CUarray dstArray, SizeT dstOffset, cuFloatComplex[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAFloatRealA(CUarray dstArray, SizeT dstOffset, cuFloatReal[] srcHost, SizeT ByteCount);
    #endregion
    #region NumberTypesArray
    delegate CUResult cuMemcpyHtoAByteA(CUarray dstArray, SizeT dstOffset, byte[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoASByteA(CUarray dstArray, SizeT dstOffset, sbyte[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAUShortA(CUarray dstArray, SizeT dstOffset, ushort[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAShortA(CUarray dstArray, SizeT dstOffset, short[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAUIntA(CUarray dstArray, SizeT dstOffset, uint[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAIntA(CUarray dstArray, SizeT dstOffset, int[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAULongA(CUarray dstArray, SizeT dstOffset, ulong[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoALongA(CUarray dstArray, SizeT dstOffset, long[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoAFloatA(CUarray dstArray, SizeT dstOffset, float[] srcHost, SizeT ByteCount);
    delegate CUResult cuMemcpyHtoADoubleA(CUarray dstArray, SizeT dstOffset, double[] srcHost, SizeT ByteCount);
    #endregion
    delegate CUResult cuMemcpyHtoAIntPtr(CUarray dstArray, SizeT dstOffset, IntPtr srcHost, SizeT ByteCount);

    #region VectorTypesArray
    delegate CUResult cuMemcpyAtoHDim3A(dim3[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHChar1A(char1[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHChar2A(char2[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHChar3A(char3[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHChar4A(char4[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHUChar1A(uchar1[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHUChar2A(uchar2[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHUChar3A(uchar3[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHUChar4A(uchar4[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHShort1A(short1[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHShort2A(short2[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHShort3A(short3[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHShort4A(short4[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHUShort1A(ushort1[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHUShort2A(ushort2[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHUShort3A(ushort3[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHUShort4A(ushort4[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHInt1A(int1[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHInt2A(int2[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHInt3A(int3[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHInt4A(int4[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHUInt1A(uint1[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHUInt2A(uint2[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHUInt3A(uint3[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHUInt4A(uint4[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHLong1A(long1[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHLong2A(long2[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHLong3A(long3[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHLong4A(long4[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHULong1A(ulong1[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHULong2A(ulong2[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHULong3A(ulong3[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHULong4A(ulong4[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHFloat1A(float1[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHFloat2A(float2[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHFloat3A(float3[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHFloat4A(float4[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHDouble1A(double1[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHDouble2A(double2[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHDoubleComplexA(cuDoubleComplex[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHDoubleRealA(cuDoubleReal[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHFloatComplexA(cuFloatComplex[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHFloatRealA(cuFloatReal[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    #endregion
    #region NumberTypesArray
    delegate CUResult cuMemcpyAtoHByteA(byte[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHSByteA(sbyte[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHUShortA(ushort[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHShortA(short[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHUIntA(uint[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHIntA(int[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHULongA(ulong[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHLongA(long[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHFloatA(float[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    delegate CUResult cuMemcpyAtoHDoubleA(double[] dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);
    #endregion

    delegate CUResult cuMemcpyAtoHIntPtr( IntPtr dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);

    // array <-> array memory
    delegate CUResult cuMemcpyAtoA(CUarray dstArray, SizeT dstOffset, CUarray srcArray, SizeT srcOffset, SizeT ByteCount);

    // 2D memcpy
    delegate CUResult cuMemcpy2D(ref CUDAMemCpy2D pCopy);
    delegate CUResult cuMemcpy2DUnaligned(ref CUDAMemCpy2D pCopy);

    // 3D memcpy
    delegate CUResult cuMemcpy3D(ref CUDAMemCpy3D pCopy);
    
    #endregion

    #region Asynchronous Memcpy
    // 1D functions
    delegate CUResult cuMemcpyHtoDAsync(CUdeviceptr dstDevice,  IntPtr srcHost, SizeT ByteCount, CUstream hStream);

    //Device -> Host
    delegate CUResult cuMemcpyDtoHAsync( IntPtr dstHost, CUdeviceptr srcDevice, SizeT ByteCount, CUstream hStream);

    // device <-> device memory
    delegate CUResult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, SizeT ByteCount, CUstream hStream);

    // system <-> array memory
    delegate CUResult cuMemcpyHtoAAsync(CUarray dstArray, SizeT dstOffset,  IntPtr srcHost, SizeT ByteCount, CUstream hStream);

    //Array -> Host
    delegate CUResult cuMemcpyAtoHAsync( IntPtr dstHost, CUarray srcArray, SizeT srcOffset, SizeT ByteCount, CUstream hStream);

    // 2D memcpy
    delegate CUResult cuMemcpy2DAsync(ref CUDAMemCpy2D pCopy, CUstream hStream);

    // 3D memcpy
    delegate CUResult cuMemcpy3DAsync(ref CUDAMemCpy3D pCopy, CUstream hStream);

    #endregion

    #region Memset
    delegate CUResult cuMemsetD8_v2(CUdeviceptr dstDevice, byte b, SizeT N);

    delegate CUResult cuMemsetD16_v2(CUdeviceptr dstDevice, ushort us, SizeT N);

    delegate CUResult cuMemsetD32_v2(CUdeviceptr dstDevice, uint ui, SizeT N);

    delegate CUResult cuMemsetD2D8_v2(CUdeviceptr dstDevice, SizeT dstPitch, byte b, SizeT Width, SizeT Height);

    delegate CUResult cuMemsetD2D16_v2(CUdeviceptr dstDevice, SizeT dstPitch, ushort us, SizeT Width, SizeT Height);

    delegate CUResult cuMemsetD2D32_v2(CUdeviceptr dstDevice, SizeT dstPitch, uint ui, SizeT Width, SizeT Height);
    
    #endregion

    #region Function management
    delegate CUResult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z);

    delegate CUResult cuFuncSetSharedSize(CUfunction hfunc, uint bytes);

    delegate CUResult cuFuncGetAttribute(ref int pi, CUFunctionAttribute attrib, CUfunction hfunc);

    delegate CUResult cuFuncSetCacheConfig(CUfunction hfunc, CUFuncCache config);
    
    #endregion

    #region Array management
    delegate CUResult cuArrayCreate_v2(ref CUarray pHandle, ref CUDAArrayDescriptor pAllocateArray);

    delegate CUResult cuArrayGetDescriptor_v2(ref CUDAArrayDescriptor pArrayDescriptor, CUarray hArray);

    delegate CUResult cuArrayDestroy(CUarray hArray);

    delegate CUResult cuArray3DCreate_v2(ref CUarray pHandle, ref CUDAArray3DDescriptor pAllocateArray);

    delegate CUResult cuArray3DGetDescriptor_v2(ref CUDAArray3DDescriptor pArrayDescriptor, CUarray hArray);
    
    #endregion

    #region Texture reference management
    delegate CUResult cuTexRefCreate(ref CUtexref pTexRef);

    delegate CUResult cuTexRefDestroy(CUtexref hTexRef);

    delegate CUResult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, CUTexRefSetArrayFlags Flags);

    delegate CUResult cuTexRefSetAddress_v2(ref SizeT ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, SizeT bytes);

    delegate CUResult cuTexRefSetAddress2D_v2(CUtexref hTexRef, ref CUDAArrayDescriptor desc, CUdeviceptr dptr, SizeT Pitch);

    delegate CUResult cuTexRefSetFormat(CUtexref hTexRef, CUArrayFormat fmt, int NumPackedComponents);

    delegate CUResult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUAddressMode am);

    delegate CUResult cuTexRefSetFilterMode(CUtexref hTexRef, CUFilterMode fm);

    delegate CUResult cuTexRefSetFlags(CUtexref hTexRef, CUTexRefSetFlags Flags);

    delegate CUResult cuTexRefGetAddress_v2(ref CUdeviceptr pdptr, CUtexref hTexRef);

    delegate CUResult cuTexRefGetArray(ref CUarray phArray, CUtexref hTexRef);

    delegate CUResult cuTexRefGetAddressMode(ref CUAddressMode pam, CUtexref hTexRef, int dim);

    delegate CUResult cuTexRefGetFilterMode(ref CUFilterMode pfm, CUtexref hTexRef);

    delegate CUResult cuTexRefGetFormat(ref CUArrayFormat pFormat, ref int pNumChannels, CUtexref hTexRef);

    delegate CUResult cuTexRefGetFlags(ref CUTexRefSetFlags pFlags, CUtexref hTexRef);
    
    #endregion

    #region Surface reference management
    delegate CUResult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, CUSurfRefSetFlags Flags);

    delegate CUResult cuSurfRefGetArray(ref CUarray phArray, CUsurfref hSurfRef);
    
    #endregion

    #region Parameter management
    delegate CUResult cuParamSetSize(CUfunction hfunc, uint numbytes);

    delegate CUResult cuParamSeti(CUfunction hfunc, int offset, uint value);

    delegate CUResult cuParamSetf(CUfunction hfunc, int offset, float value);

    delegate CUResult cuParamSetvIntPtr(CUfunction hfunc, int offset, IntPtr ptr, uint numbytes);
    delegate CUResult cuParamSetvByte(CUfunction hfunc, int offset, ref byte ptr, uint numbytes);
    delegate CUResult cuParamSetvSByte(CUfunction hfunc, int offset, ref sbyte ptr, uint numbytes);
    delegate CUResult cuParamSetvUShort(CUfunction hfunc, int offset, ref ushort ptr, uint numbytes);
    delegate CUResult cuParamSetvShort(CUfunction hfunc, int offset, ref short ptr, uint numbytes);
    delegate CUResult cuParamSetvUInt(CUfunction hfunc, int offset, ref uint ptr, uint numbytes);
    delegate CUResult cuParamSetvInt(CUfunction hfunc, int offset, ref int ptr, uint numbytes);
    delegate CUResult cuParamSetvULong(CUfunction hfunc, int offset, ref ulong ptr, uint numbytes);
    delegate CUResult cuParamSetvLong(CUfunction hfunc, int offset, ref long ptr, uint numbytes);
    delegate CUResult cuParamSetvFloat(CUfunction hfunc, int offset, ref float ptr, uint numbytes);
    delegate CUResult cuParamSetvDouble(CUfunction hfunc, int offset, ref double ptr, uint numbytes);

    #region VectorTypes
    delegate CUResult cuParamSetvDim3(CUfunction hfunc, int offset, ref dim3 ptr, uint numbytes);
    delegate CUResult cuParamSetvChar1(CUfunction hfunc, int offset, ref char1 ptr, uint numbytes);
    delegate CUResult cuParamSetvChar2(CUfunction hfunc, int offset, ref char2 ptr, uint numbytes);
    delegate CUResult cuParamSetvChar3(CUfunction hfunc, int offset, ref char3 ptr, uint numbytes);
    delegate CUResult cuParamSetvChar4(CUfunction hfunc, int offset, ref char4 ptr, uint numbytes);
    delegate CUResult cuParamSetvUChar1(CUfunction hfunc, int offset, ref uchar1 ptr, uint numbytes);
    delegate CUResult cuParamSetvUChar2(CUfunction hfunc, int offset, ref uchar2 ptr, uint numbytes);
    delegate CUResult cuParamSetvUChar3(CUfunction hfunc, int offset, ref uchar3 ptr, uint numbytes);
    delegate CUResult cuParamSetvUChar4(CUfunction hfunc, int offset, ref uchar4 ptr, uint numbytes);
    delegate CUResult cuParamSetvShort1(CUfunction hfunc, int offset, ref short1 ptr, uint numbytes);
    delegate CUResult cuParamSetvShort2(CUfunction hfunc, int offset, ref short2 ptr, uint numbytes);
    delegate CUResult cuParamSetvShort3(CUfunction hfunc, int offset, ref short3 ptr, uint numbytes);
    delegate CUResult cuParamSetvShort4(CUfunction hfunc, int offset, ref short4 ptr, uint numbytes);
    delegate CUResult cuParamSetvUShort1(CUfunction hfunc, int offset, ref ushort1 ptr, uint numbytes);
    delegate CUResult cuParamSetvUShort2(CUfunction hfunc, int offset, ref ushort2 ptr, uint numbytes);
    delegate CUResult cuParamSetvUShort3(CUfunction hfunc, int offset, ref ushort3 ptr, uint numbytes);
    delegate CUResult cuParamSetvUShort4(CUfunction hfunc, int offset, ref ushort4 ptr, uint numbytes);
    delegate CUResult cuParamSetvInt1(CUfunction hfunc, int offset, ref int1 ptr, uint numbytes);
    delegate CUResult cuParamSetvInt2(CUfunction hfunc, int offset, ref int2 ptr, uint numbytes);
    delegate CUResult cuParamSetvInt3(CUfunction hfunc, int offset, ref int3 ptr, uint numbytes);
    delegate CUResult cuParamSetvInt4(CUfunction hfunc, int offset, ref int4 ptr, uint numbytes);
    delegate CUResult cuParamSetvUInt1(CUfunction hfunc, int offset, ref uint1 ptr, uint numbytes);
    delegate CUResult cuParamSetvUInt2(CUfunction hfunc, int offset, ref uint2 ptr, uint numbytes);
    delegate CUResult cuParamSetvUInt3(CUfunction hfunc, int offset, ref uint3 ptr, uint numbytes);
    delegate CUResult cuParamSetvUInt4(CUfunction hfunc, int offset, ref uint4 ptr, uint numbytes);
    delegate CUResult cuParamSetvLong1(CUfunction hfunc, int offset, ref long1 ptr, uint numbytes);
    delegate CUResult cuParamSetvLong2(CUfunction hfunc, int offset, ref long2 ptr, uint numbytes);
    delegate CUResult cuParamSetvLong3(CUfunction hfunc, int offset, ref long3 ptr, uint numbytes);
    delegate CUResult cuParamSetvLong4(CUfunction hfunc, int offset, ref long4 ptr, uint numbytes);
    delegate CUResult cuParamSetvULong1(CUfunction hfunc, int offset, ref ulong1 ptr, uint numbytes);
    delegate CUResult cuParamSetvULong2(CUfunction hfunc, int offset, ref ulong2 ptr, uint numbytes);
    delegate CUResult cuParamSetvULong3(CUfunction hfunc, int offset, ref ulong3 ptr, uint numbytes);
    delegate CUResult cuParamSetvULong4(CUfunction hfunc, int offset, ref ulong4 ptr, uint numbytes);
    delegate CUResult cuParamSetvFloat1(CUfunction hfunc, int offset, ref float1 ptr, uint numbytes);
    delegate CUResult cuParamSetvFloat2(CUfunction hfunc, int offset, ref float2 ptr, uint numbytes);
    delegate CUResult cuParamSetvFloat3(CUfunction hfunc, int offset, ref float3 ptr, uint numbytes);
    delegate CUResult cuParamSetvFloat4(CUfunction hfunc, int offset, ref float4 ptr, uint numbytes);
    delegate CUResult cuParamSetvDouble1(CUfunction hfunc, int offset, ref double1 ptr, uint numbytes);
    delegate CUResult cuParamSetvDouble2(CUfunction hfunc, int offset, ref double2 ptr, uint numbytes);
    delegate CUResult cuParamSetvDoubleComplex(CUfunction hfunc, int offset, ref cuDoubleComplex ptr, uint numbytes);
    delegate CUResult cuParamSetvDoubleReal(CUfunction hfunc, int offset, ref cuDoubleReal ptr, uint numbytes);
    delegate CUResult cuParamSetvFloatComplex(CUfunction hfunc, int offset, ref cuFloatComplex ptr, uint numbytes);
    delegate CUResult cuParamSetvFloatReal(CUfunction hfunc, int offset, ref cuFloatReal ptr, uint numbytes);
    #endregion

    #region VectorTypesArrays
    delegate CUResult cuParamSetvDim3A(CUfunction hfunc, int offset, dim3[] ptr, uint numbytes);
    delegate CUResult cuParamSetvChar1A(CUfunction hfunc, int offset, char1[] ptr, uint numbytes);
    delegate CUResult cuParamSetvChar2A(CUfunction hfunc, int offset, char2[] ptr, uint numbytes);
    delegate CUResult cuParamSetvChar3A(CUfunction hfunc, int offset, char3[] ptr, uint numbytes);
    delegate CUResult cuParamSetvChar4A(CUfunction hfunc, int offset, char4[] ptr, uint numbytes);
    delegate CUResult cuParamSetvUChar1A(CUfunction hfunc, int offset, uchar1[] ptr, uint numbytes);
    delegate CUResult cuParamSetvUChar2A(CUfunction hfunc, int offset, uchar2[] ptr, uint numbytes);
    delegate CUResult cuParamSetvUChar3A(CUfunction hfunc, int offset, uchar3[] ptr, uint numbytes);
    delegate CUResult cuParamSetvUChar4A(CUfunction hfunc, int offset, uchar4[] ptr, uint numbytes);
    delegate CUResult cuParamSetvShort1A(CUfunction hfunc, int offset, short1[] ptr, uint numbytes);
    delegate CUResult cuParamSetvShort2A(CUfunction hfunc, int offset, short2[] ptr, uint numbytes);
    delegate CUResult cuParamSetvShort3A(CUfunction hfunc, int offset, short3[] ptr, uint numbytes);
    delegate CUResult cuParamSetvShort4A(CUfunction hfunc, int offset, short4[] ptr, uint numbytes);
    delegate CUResult cuParamSetvUShort1A(CUfunction hfunc, int offset, ushort1[] ptr, uint numbytes);
    delegate CUResult cuParamSetvUShort2A(CUfunction hfunc, int offset, ushort2[] ptr, uint numbytes);
    delegate CUResult cuParamSetvUShort3A(CUfunction hfunc, int offset, ushort3[] ptr, uint numbytes);
    delegate CUResult cuParamSetvUShort4A(CUfunction hfunc, int offset, ushort4[] ptr, uint numbytes);
    delegate CUResult cuParamSetvInt1A(CUfunction hfunc, int offset, int1[] ptr, uint numbytes);
    delegate CUResult cuParamSetvInt2A(CUfunction hfunc, int offset, int2[] ptr, uint numbytes);
    delegate CUResult cuParamSetvInt3A(CUfunction hfunc, int offset, int3[] ptr, uint numbytes);
    delegate CUResult cuParamSetvInt4A(CUfunction hfunc, int offset, int4[] ptr, uint numbytes);
    delegate CUResult cuParamSetvUInt1A(CUfunction hfunc, int offset, uint1[] ptr, uint numbytes);
    delegate CUResult cuParamSetvUInt2A(CUfunction hfunc, int offset, uint2[] ptr, uint numbytes);
    delegate CUResult cuParamSetvUInt3A(CUfunction hfunc, int offset, uint3[] ptr, uint numbytes);
    delegate CUResult cuParamSetvUInt4A(CUfunction hfunc, int offset, uint4[] ptr, uint numbytes);
    delegate CUResult cuParamSetvLong1A(CUfunction hfunc, int offset, long1[] ptr, uint numbytes);
    delegate CUResult cuParamSetvLong2A(CUfunction hfunc, int offset, long2[] ptr, uint numbytes);
    delegate CUResult cuParamSetvLong3A(CUfunction hfunc, int offset, long3[] ptr, uint numbytes);
    delegate CUResult cuParamSetvLong4A(CUfunction hfunc, int offset, long4[] ptr, uint numbytes);
    delegate CUResult cuParamSetvULong1A(CUfunction hfunc, int offset, ulong1[] ptr, uint numbytes);
    delegate CUResult cuParamSetvULong2A(CUfunction hfunc, int offset, ulong2[] ptr, uint numbytes);
    delegate CUResult cuParamSetvULong3A(CUfunction hfunc, int offset, ulong3[] ptr, uint numbytes);
    delegate CUResult cuParamSetvULong4A(CUfunction hfunc, int offset, ulong4[] ptr, uint numbytes);
    delegate CUResult cuParamSetvFloat1A(CUfunction hfunc, int offset, float1[] ptr, uint numbytes);
    delegate CUResult cuParamSetvFloat2A(CUfunction hfunc, int offset, float2[] ptr, uint numbytes);
    delegate CUResult cuParamSetvFloat3A(CUfunction hfunc, int offset, float3[] ptr, uint numbytes);
    delegate CUResult cuParamSetvFloat4A(CUfunction hfunc, int offset, float4[] ptr, uint numbytes);
    delegate CUResult cuParamSetvDouble1A(CUfunction hfunc, int offset, double1[] ptr, uint numbytes);
    delegate CUResult cuParamSetvDouble2A(CUfunction hfunc, int offset, double2[] ptr, uint numbytes);
    delegate CUResult cuParamSetvDoubleComplexA(CUfunction hfunc, int offset, cuDoubleComplex[] ptr, uint numbytes);
    delegate CUResult cuParamSetvDoubleRealA(CUfunction hfunc, int offset, cuDoubleReal[] ptr, uint numbytes);
    delegate CUResult cuParamSetvFloatComplexA(CUfunction hfunc, int offset, cuFloatComplex[] ptr, uint numbytes);
    delegate CUResult cuParamSetvFloatRealA(CUfunction hfunc, int offset, cuFloatReal[] ptr, uint numbytes);
    #endregion

    delegate CUResult cuParamSetvByteA(CUfunction hfunc, int offset, byte[] ptr, uint numbytes);
    delegate CUResult cuParamSetvSByteA(CUfunction hfunc, int offset, sbyte[] ptr, uint numbytes);
    delegate CUResult cuParamSetvUShortA(CUfunction hfunc, int offset, ushort[] ptr, uint numbytes);
    delegate CUResult cuParamSetvShortA(CUfunction hfunc, int offset, short[] ptr, uint numbytes);
    delegate CUResult cuParamSetvUIntA(CUfunction hfunc, int offset, uint[] ptr, uint numbytes);
    delegate CUResult cuParamSetvIntA(CUfunction hfunc, int offset, int[] ptr, uint numbytes);
    delegate CUResult cuParamSetvULongA(CUfunction hfunc, int offset, ulong[] ptr, uint numbytes);
    delegate CUResult cuParamSetvLongA(CUfunction hfunc, int offset, long[] ptr, uint numbytes);
    delegate CUResult cuParamSetvFloatA(CUfunction hfunc, int offset, float[] ptr, uint numbytes);
    delegate CUResult cuParamSetvDoubleA(CUfunction hfunc, int offset, double[] ptr, uint numbytes);

    delegate CUResult cuParamSetTexRef(CUfunction hfunc, CUParameterTexRef texunit, CUtexref hTexRef);
    
    #endregion

    #region Launch functions
    delegate CUResult cuLaunch(CUfunction f);

    delegate CUResult cuLaunchGrid(CUfunction f, int grid_width, int grid_height);

    delegate CUResult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream);
    
    #endregion

    #region Events
    delegate CUResult cuEventCreate(ref CUevent phEvent, CUEventFlags Flags);

    delegate CUResult cuEventRecord(CUevent hEvent, CUstream hStream);

    delegate CUResult cuEventQuery(CUevent hEvent);

    delegate CUResult cuEventSynchronize(CUevent hEvent);

    delegate CUResult cuEventDestroy(CUevent hEvent);

    delegate CUResult cuEventElapsedTime(ref float pMilliseconds, CUevent hStart, CUevent hEnd);
    
    #endregion

    #region Streams
    delegate CUResult cuStreamCreate(ref CUstream phStream, CUStreamFlags Flags);

    delegate CUResult cuStreamQuery(CUstream hStream);

    delegate CUResult cuStreamSynchronize(CUstream hStream);

    delegate CUResult cuStreamDestroy(CUstream hStream);
    
    #endregion

    #region Graphics interop
    delegate CUResult cuGraphicsUnregisterResource(CUgraphicsResource resource);

    delegate CUResult cuGraphicsSubResourceGetMappedArray(ref CUarray pArray, CUgraphicsResource resource, uint arrayIndex, uint mipLevel);

    delegate CUResult cuGraphicsResourceGetMappedPointer_v2(ref CUdeviceptr pDevPtr, ref SizeT pSize, CUgraphicsResource resource);

    delegate CUResult cuGraphicsResourceSetMapFlags(CUgraphicsResource resource, CUGraphicsMapResourceFlags flags);

    delegate CUResult cuGraphicsMapResources(uint count, object resources, CUstream hStream);
    //delegate CUResult cuGraphicsMapResources(uint count, ref CUgraphicsResource resources, CUstream hStream);

    //delegate CUResult cuGraphicsMapResources(uint count, CUgraphicsResource[] resources, CUstream hStream);

    delegate CUResult cuGraphicsUnmapResources(uint count, object resources, CUstream hStream);
    //delegate CUResult cuGraphicsUnmapResources(uint count, ref CUgraphicsResource resources, CUstream hStream);

    //delegate CUResult cuGraphicsUnmapResources(uint count, CUgraphicsResource[] resources, CUstream hStream);
    
    #endregion

    #region Export tables
    delegate CUResult cuGetExportTable(ref IntPtr ppExportTable, ref CUuuid pExportTableId);
    
    #endregion

    #region Limits
    delegate CUResult cuCtxSetLimit(CULimit limit, SizeT value);

    delegate CUResult cuCtxGetLimit(ref SizeT pvalue, CULimit limit);
    
    #endregion

    #region DirectX
    delegate CUResult cuD3D9GetDevice(ref CUdevice pCudaDevice, string pszAdapterName);
    delegate CUResult cuD3D9CtxCreate(ref CUcontext pCtx, ref CUdevice pCudaDevice, CUCtxFlags Flags, IntPtr pD3DDevice);
    delegate CUResult cuGraphicsD3D9RegisterResource(ref CUgraphicsResource pCudaResource, IntPtr pD3DResource, CUGraphicsRegisterFlags Flags);
    delegate CUResult cuD3D9GetDirect3DDevice(ref IntPtr ppD3DDevice);
    //delegate CUResult cuD3D9UnregisterResource(IntPtr pResource);
    //delegate CUResult cuD3D9MapResources(uint count, ref IntPtr ppResource);
    //delegate CUResult cuD3D9UnmapResources(uint count, ref IntPtr ppResource);
    //delegate CUResult cuD3D9ResourceGetSurfaceDimensions(ref uint pWidth, ref uint pHeight, ref uint pDepth, IntPtr pResource, uint Face, uint Level);
    //delegate CUResult cuD3D9ResourceGetMappedArray(ref CUarray pArray, IntPtr pResource, uint Face, uint Level);
    //delegate CUResult cuD3D9ResourceGetMappedPointer(ref CUdeviceptr pDevPtr, IntPtr pResource, uint Face, uint Level);
    //delegate CUResult cuD3D9ResourceGetMappedSize(ref uint pSize, IntPtr pResource, uint Face, uint Level);
    //delegate CUResult cuD3D9ResourceGetMappedPitch(ref uint pPitch, ref uint pPitchSlice, IntPtr pResource, uint Face, uint Level);
    //delegate CUResult cuD3D9Begin(IntPtr pDevice);
    //delegate CUResult cuD3D9End();
    //delegate CUResult cuD3D9RegisterVertexBuffer(IntPtr pVB);
    //delegate CUResult cuD3D9MapVertexBuffer(ref CUdeviceptr pDevPtr, ref uint pSize, IntPtr pVB);
    //delegate CUResult cuD3D9UnmapVertexBuffer(IntPtr pVB);
    //delegate CUResult cuD3D9UnregisterVertexBuffer(IntPtr pVB);
    delegate CUResult cuD3D10GetDevice(ref CUdevice device, IntPtr pAdapter);
    delegate CUResult cuD3D10CtxCreate(ref CUcontext pCtx, ref CUdevice pCudaDevice, CUCtxFlags Flags, IntPtr pD3DDevice);
    delegate CUResult cuGraphicsD3D10RegisterResource(ref CUgraphicsResource pCudaResource, IntPtr pD3DResource, CUGraphicsRegisterFlags Flags);
    //delegate CUResult cuD3D10UnregisterResource(IntPtr pResource);
    //delegate CUResult cuD3D10MapResources(uint count, ref IntPtr ppResource);
    //delegate CUResult cuD3D10UnmapResources(uint count, ref IntPtr ppResource);
    //delegate CUResult cuD3D10ResourceGetSurfaceDimensions(ref uint pWidth, ref uint pHeight, ref uint pDepth, IntPtr pResource, uint Face, uint Level);
    //delegate CUResult cuD3D10ResourceGetMappedArray(ref CUarray pArray, IntPtr pResource, uint Face, uint Level);
    //delegate CUResult cuD3D10ResourceGetMappedPointer(ref CUdeviceptr pDevPtr, IntPtr pResource, uint Face, uint Level);
    //delegate CUResult cuD3D10ResourceGetMappedSize(ref uint pSize, IntPtr pResource, uint Face, uint Level);
    //delegate CUResult cuD3D10ResourceGetMappedPitch(ref uint pPitch, ref uint pPitchSlice, IntPtr pResource, uint Face, uint Level);
    delegate CUResult cuD3D11GetDevice(ref CUdevice device, IntPtr pAdapter);
    delegate CUResult cuD3D11CtxCreate(ref CUcontext pCtx, ref CUdevice pCudaDevice, CUCtxFlags Flags, IntPtr pD3DDevice);
    delegate CUResult cuGraphicsD3D11RegisterResource(ref CUgraphicsResource pCudaResource, IntPtr pD3DResource, CUGraphicsRegisterFlags Flags);

    #endregion

    #region OpenGL
    delegate CUResult cuGLCtxCreate(ref CUcontext pCtx, CUCtxFlags Flags, CUdevice device);
    delegate CUResult cuGraphicsGLRegisterBuffer(ref CUgraphicsResource pCudaResource, uint buffer, CUGraphicsRegisterFlags Flags);
    delegate CUResult cuGraphicsGLRegisterImage(ref CUgraphicsResource pCudaResource, uint image, ulong target, CUGraphicsRegisterFlags Flags);
    delegate CUResult cuWGLGetDevice(ref CUdevice pDevice, IntPtr hGpu);
    #endregion
}
