//	Copyright (c) 2012, Michael Kunz. All rights reserved.
//	http://managedcuda.codeplex.com
//
//	This file is part of ManagedCuda.
//
//	ManagedCuda is free software: you can redistribute it and/or modify
//	it under the terms of the GNU Lesser General Public License as 
//	published by the Free Software Foundation, either version 2.1 of the 
//	License, or (at your option) any later version.
//
//	ManagedCuda is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//	GNU Lesser General Public License for more details.
//
//	You should have received a copy of the GNU Lesser General Public
//	License along with this library; if not, write to the Free Software
//	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
//	MA 02110-1301  USA, http://www.gnu.org/licenses/.


using System;
using System.Collections.Generic;
using System.Text;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Runtime.InteropServices;
using System.Diagnostics;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;

namespace ManagedCuda.CudaBitmapSource
{
	/// <summary>
	/// CudaBitmapSource is more like a proof of concept and work in progress. 
	/// I'm not yet satisfied of it's realisation, but it works quiet nice.
	/// </summary>
	public class CudaBitmapSource : BitmapSource, IDisposable
	{
		private WriteableBitmap _hostBitmap;
		private CUdeviceptr _deviceBitmap; 
		private int _channelCount;
		private int _channelWidth;
		private int _bitsPerPixel;
		private SizeT _bytesPerPixel;
		private bool _usePitched;
		private int  _hostStride;
		private SizeT _deviceStride;
		private CUResult res;
		private bool disposed;

		#region Constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="_source"></param>
		public CudaBitmapSource(BitmapSource _source)
		{
			_usePitched = CheckPixelFormat(_source.Format);

			//alloc host
			_hostBitmap = new WriteableBitmap(_source);
			_hostStride = _hostBitmap.BackBufferStride;

			_channelWidth = GetChannelWidth(_source.Format);
			_channelCount = GetChannelCount(_source.Format);
			_bitsPerPixel = _channelCount * _channelWidth;
			_bytesPerPixel = _bitsPerPixel / 8;

			//alloc Device
			AllocDevice(_source.PixelWidth, _source.PixelHeight);

			AttachEvents();

			UpdateDevice();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="pixelWidth"></param>
		/// <param name="pixelHeight"></param>
		/// <param name="dpiX"></param>
		/// <param name="dpiY"></param>
		/// <param name="pixelFormat"></param>
		/// <param name="palette"></param>
		public CudaBitmapSource(int pixelWidth, int pixelHeight, double dpiX, double dpiY, PixelFormat pixelFormat, BitmapPalette palette)
		{
			_usePitched = CheckPixelFormat(pixelFormat);

			//alloc host
			_hostBitmap = new WriteableBitmap(pixelWidth, pixelHeight, dpiX, dpiY, pixelFormat, palette);
			_hostStride = _hostBitmap.BackBufferStride;

			_channelWidth = GetChannelWidth(pixelFormat);
			_channelCount = GetChannelCount(pixelFormat);
			_bitsPerPixel = _channelCount * _channelWidth;
			_bytesPerPixel = _bitsPerPixel / 8;

			//alloc Device
			AllocDevice(pixelWidth, pixelHeight);

			AttachEvents();
		}     
		

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaBitmapSource()
		{
			Dispose (false);
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
		protected virtual void Dispose (bool fDisposing)
		{
		   if (fDisposing && !disposed)
		   {
			   //Ignore if failing
			   res = DriverAPINativeMethods.MemoryManagement.cuMemFree_v2(_deviceBitmap);
			   Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemFree", res));
			   disposed = true;
		   }
		   if (!fDisposing && !disposed)
			   Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region private helper methods
		private void AllocDevice(int pixelWidth, int pixelHeight)
		{
			//alloc Device
			if (_usePitched)
			{
				_deviceBitmap = new CUdeviceptr();

				res = DriverAPINativeMethods.MemoryManagement.cuMemAllocPitch_v2(ref _deviceBitmap, ref _deviceStride, _bytesPerPixel * pixelWidth, pixelHeight, _bytesPerPixel);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Pitch is: {3}", DateTime.Now, "cuMemAllocPitch", res, _deviceStride));
				if (res != CUResult.Success) throw new CudaException(res);
			}
			else
			{
				_deviceBitmap = new CUdeviceptr();
				_deviceStride = _hostBitmap.BackBufferStride;
				res = DriverAPINativeMethods.MemoryManagement.cuMemAlloc_v2(ref _deviceBitmap, _hostBitmap.BackBufferStride * pixelHeight);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAlloc", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
		}

		/// <summary>
		/// Get the number of color channels for a given pixel format
		/// </summary>
		/// <param name="pixelFormat"></param>
		/// <returns>0 if not supported, #channels otherwise</returns>
		private int GetChannelCount(PixelFormat pixelFormat)
		{
			if (pixelFormat == PixelFormats.Bgr101010) return 3;
			if (pixelFormat == PixelFormats.Bgr24) return 3;
			if (pixelFormat == PixelFormats.Bgr32) return 4; //alpha is unused
			if (pixelFormat == PixelFormats.Bgr555) return 3;
			if (pixelFormat == PixelFormats.Bgr565) return 3;
			if (pixelFormat == PixelFormats.Bgra32) return 4;
			if (pixelFormat == PixelFormats.BlackWhite) return 1;
			if (pixelFormat == PixelFormats.Cmyk32) return 4;
			if (pixelFormat == PixelFormats.Gray16) return 1;
			if (pixelFormat == PixelFormats.Gray2) return 1;
			if (pixelFormat == PixelFormats.Gray32Float) return 1;
			if (pixelFormat == PixelFormats.Gray4) return 1;
			if (pixelFormat == PixelFormats.Gray8) return 1;
			if (pixelFormat == PixelFormats.Indexed1) return 1;
			if (pixelFormat == PixelFormats.Indexed2) return 1;
			if (pixelFormat == PixelFormats.Indexed4) return 1;
			if (pixelFormat == PixelFormats.Indexed8) return 1;
			if (pixelFormat == PixelFormats.Pbgra32) return 4;
			if (pixelFormat == PixelFormats.Prgba128Float) return 4;
			if (pixelFormat == PixelFormats.Prgba64) return 4; //Documentation says 32, but that doesn't sum up to 64
			if (pixelFormat == PixelFormats.Rgb128Float) return 4; //alpha is unused;
			if (pixelFormat == PixelFormats.Rgb24) return 3;
			if (pixelFormat == PixelFormats.Rgb48) return 3;
			if (pixelFormat == PixelFormats.Rgba128Float) return 4;
			if (pixelFormat == PixelFormats.Rgba64) return 4;

			return 0;
		}

		/// <summary>
		/// Get the size of one pixel for one channel in bits
		/// </summary>
		/// <param name="pixelFormat"></param>
		/// <returns>0 if not supported, #Bits otherwise</returns>
		private int GetChannelWidth(PixelFormat pixelFormat)
		{
			if (pixelFormat == PixelFormats.Bgr101010) return 10;
			if (pixelFormat == PixelFormats.Bgr24) return 8;
			if (pixelFormat == PixelFormats.Bgr32) return 8;
			if (pixelFormat == PixelFormats.Bgr555) return 5;
			if (pixelFormat == PixelFormats.Bgr565) return 0;
			if (pixelFormat == PixelFormats.Bgra32) return 8;
			if (pixelFormat == PixelFormats.BlackWhite) return 1;
			if (pixelFormat == PixelFormats.Cmyk32) return 8;
			if (pixelFormat == PixelFormats.Gray16) return 16;
			if (pixelFormat == PixelFormats.Gray2) return 2;
			if (pixelFormat == PixelFormats.Gray32Float) return 32;
			if (pixelFormat == PixelFormats.Gray4) return 4;
			if (pixelFormat == PixelFormats.Gray8) return 8;
			if (pixelFormat == PixelFormats.Indexed1) return 1;
			if (pixelFormat == PixelFormats.Indexed2) return 2;
			if (pixelFormat == PixelFormats.Indexed4) return 4;
			if (pixelFormat == PixelFormats.Indexed8) return 8;
			if (pixelFormat == PixelFormats.Pbgra32) return 8;
			if (pixelFormat == PixelFormats.Prgba128Float) return 32;
			if (pixelFormat == PixelFormats.Prgba64) return 16; //Documentation says 32, but that doesn't sum up to 64
			if (pixelFormat == PixelFormats.Rgb128Float) return 32;
			if (pixelFormat == PixelFormats.Rgb24) return 8;
			if (pixelFormat == PixelFormats.Rgb48) return 16;
			if (pixelFormat == PixelFormats.Rgba128Float) return 32;
			if (pixelFormat == PixelFormats.Rgba64) return 16;
			
			return 0;
		}

		/// <summary>
		/// Returns the size of one pixel in bits
		/// </summary>
		/// <param name="pixelFormat"></param>
		/// <returns></returns>
		private int GetPixelSize(PixelFormat pixelFormat)
		{
			if (pixelFormat == PixelFormats.Bgr565) return 16;
			int pixelSize = GetChannelCount(pixelFormat) * GetChannelWidth(pixelFormat);
			return pixelSize;
		}

		/// <summary>
		/// Checks if the pixel format is supported by CudaPitchedVariable.
		/// </summary>
		/// <param name="pixelFormat">Pixelformat to check</param>
		/// <returns>True if the pixelformat is supported</returns>
		private bool CheckPixelFormat(PixelFormat pixelFormat)
		{
			int pixelSize = GetChannelCount(pixelFormat) * GetChannelWidth(pixelFormat);

			if (pixelSize == 0 || pixelSize > 128) return false; //format is not listed or larger than 128 bits

			if (pixelSize % (16 * 8) == 0) return true;
			if (pixelSize % (8 * 8) == 0) return true;
			if (pixelSize % (4 * 8) == 0) return true;

			return false;
		}

		private void AttachEvents()
		{
			_hostBitmap.DecodeFailed += new EventHandler<ExceptionEventArgs>(_hostBitmap_DecodeFailed);
			_hostBitmap.DownloadCompleted += new EventHandler(_hostBitmap_DownloadCompleted);
			_hostBitmap.DownloadFailed += new EventHandler<ExceptionEventArgs>(_hostBitmap_DownloadFailed);
			_hostBitmap.DownloadProgress += new EventHandler<DownloadProgressEventArgs>(_hostBitmap_DownloadProgress);
			_hostBitmap.Changed += new EventHandler(_hostBitmap_Changed);            
		}
		#endregion
		
		#region Override BitmapSource properties
		/// <summary>
		/// Gets the horizontal dots per inch (dpi) of the image.
		/// </summary>
		public override double DpiX
		{
			get { return _hostBitmap.DpiX; }
		}

		/// <summary>
		/// Gets the vertical dots per inch (dpi) of the image. 
		/// </summary>
		public override double DpiY
		{
			get { return _hostBitmap.DpiY; }
		}

		/// <summary>
		/// Gets the native PixelFormat of the bitmap data.
		/// </summary>
		public override PixelFormat Format
		{
			get { return _hostBitmap.Format; }
		}

		/// <summary>
		/// Gets the height of the source bitmap in device-independent units (1/96th inch per unit). (Overrides ImageSource.Height.)
		/// </summary>
		public override double Height
		{
			get { return _hostBitmap.Height; }
		}

		/// <summary>
		/// Gets a value that indicates whether the BitmapSource content is currently downloading. 
		/// </summary>
		public override bool IsDownloading
		{
			get { return _hostBitmap.IsDownloading; }
		}

		/// <summary>
		/// Gets the metadata that is associated with this bitmap image. (Overrides ImageSource.Metadata.)
		/// </summary>
		public override ImageMetadata Metadata
		{
			get { return _hostBitmap.Metadata; }
		}

		/// <summary>
		/// Gets the color palette of the bitmap, if one is specified. 
		/// </summary>
		public override BitmapPalette Palette
		{
			get { return _hostBitmap.Palette; }
		}

		/// <summary>
		/// Gets the height of the bitmap in pixels.
		/// </summary>
		public override int PixelHeight
		{
			get { return _hostBitmap.PixelHeight; }
		}

		/// <summary>
		/// Gets the width of the bitmap in pixels.
		/// </summary>
		public override int PixelWidth
		{
			get { return _hostBitmap.PixelWidth; }
		}

		/// <summary>
		/// Gets the width of the bitmap in device-independent units (1/96th inch per unit). (Overrides ImageSource.Width.)
		/// </summary>
		public override double Width
		{
			get { return _hostBitmap.Width; }
		}
		#endregion

		#region Override BitmapSource methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="sourceFreezable"></param>
		protected override void CloneCore(Freezable sourceFreezable)
		{
			base.CloneCore(sourceFreezable);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="sourceFreezable"></param>
		protected override void CloneCurrentValueCore(Freezable sourceFreezable)
		{
			base.CloneCurrentValueCore(sourceFreezable);
		}

		/// <summary>
		/// Copies the bitmap pixel data into an array of pixels with the specified stride, 
		/// starting at the specified offset. (Host only)
		/// </summary>
		/// <param name="pixels">The destination array.</param>
		/// <param name="stride">The stride of the bitmap.</param>
		/// <param name="offset">The pixel location where copying starts.</param>
		public override void CopyPixels(Array pixels, int stride, int offset)
		{
			_hostBitmap.CopyPixels(pixels, stride, offset);
		}

		/// <summary>
		/// Copies the bitmap pixel data within the specified rectangle into an array of pixels that has the 
		/// specified stride starting at the specified offset. (Host only)
		/// </summary>
		/// <param name="sourceRect">The source rectangle to copy.An <see cref="Int32Rect.Empty"/> value specifies the entire bitmap.</param>
		/// <param name="pixels">The destination array.</param>
		/// <param name="stride">The stride of the bitmap.</param>
		/// <param name="offset">The pixel location where copying starts.</param>
		public override void CopyPixels(System.Windows.Int32Rect sourceRect, Array pixels, int stride, int offset)
		{
			_hostBitmap.CopyPixels(sourceRect, pixels, stride, offset);
		}

		/// <summary>
		/// Copies the bitmap pixel data within the specified rectangle. (Host only)
		/// </summary>
		/// <param name="sourceRect">The source rectangle to copy.An <see cref="Int32Rect.Empty"/> value specifies the entire bitmap.</param>
		/// <param name="buffer">A pointer to the buffer.</param>
		/// <param name="bufferSize">The size of the buffer.</param>
		/// <param name="stride">The stride of the bitmap.</param>
		public override void CopyPixels(System.Windows.Int32Rect sourceRect, IntPtr buffer, int bufferSize, int stride)
		{
			_hostBitmap.CopyPixels(sourceRect, buffer, bufferSize, stride);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		protected override Freezable CreateInstanceCore()
		{
			return (Freezable) Activator.CreateInstance(this.GetType());
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="isChecking"></param>
		/// <returns></returns>
		protected override bool FreezeCore(bool isChecking)
		{
			return base.FreezeCore(isChecking);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="sourceFreezable"></param>
		protected override void GetAsFrozenCore(System.Windows.Freezable sourceFreezable)
		{
			base.GetAsFrozenCore(sourceFreezable);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="sourceFreezable"></param>
		protected override void GetCurrentValueAsFrozenCore(System.Windows.Freezable sourceFreezable)
		{
			base.GetCurrentValueAsFrozenCore(sourceFreezable);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="e"></param>
		protected override void OnPropertyChanged(System.Windows.DependencyPropertyChangedEventArgs e)
		{            
			base.OnPropertyChanged(e);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="dp"></param>
		/// <returns></returns>
		protected override bool ShouldSerializeProperty(System.Windows.DependencyProperty dp)
		{
			return base.ShouldSerializeProperty(dp);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return _hostBitmap.ToString();
		}
		#endregion

		#region Override BitmapSource events
		/// <summary>
		/// Occurs when the image fails to load, due to a corrupt image header. (Inherited from BitmapSource.)
		/// </summary>
		public override event EventHandler<ExceptionEventArgs> DecodeFailed;
		/// <summary>
		/// Occurs when the bitmap content has been completely downloaded. (Inherited from BitmapSource.)
		/// </summary>
		public override event EventHandler DownloadCompleted;
		/// <summary>
		/// Occurs when the bitmap content failed to download. (Inherited from BitmapSource.)
		/// </summary>
		public override event EventHandler<ExceptionEventArgs> DownloadFailed;
		/// <summary>
		/// Occurs when the download progress of the bitmap content has changed. (Inherited from BitmapSource.)
		/// </summary>
		public override event EventHandler<DownloadProgressEventArgs> DownloadProgress;
		
		void _hostBitmap_DecodeFailed(object sender, ExceptionEventArgs e)
		{
			if (this.DecodeFailed != null)
				this.DecodeFailed(this, e);
		}

		void _hostBitmap_DownloadCompleted(object sender, EventArgs e)
		{
			if (this.DownloadCompleted != null)
				this.DownloadCompleted(this, e);
		}

		void _hostBitmap_DownloadFailed(object sender, ExceptionEventArgs e)
		{
			if (this.DownloadFailed != null)
				this.DownloadFailed(this, e);
		}

		void _hostBitmap_DownloadProgress(object sender, DownloadProgressEventArgs e)
		{
			if (this.DownloadProgress != null)
				this.DownloadProgress(this, e);
		}

		void _hostBitmap_Changed(object sender, EventArgs e)
		{
			//if (this.Changed != null)
			//    Changed(sender, e);
		}
		#endregion

		#region Properties WriteableBitmap
		/// <summary>
		/// Gets a pointer to the back buffer. 
		/// </summary>
		public IntPtr BackBuffer
		{
			get { return _hostBitmap.BackBuffer; }
		}

		/// <summary>
		/// Gets a value indicating the number of bytes in a single row of pixel data. 
		/// </summary>
		public int BackBufferStride
		{
			get { return _hostBitmap.BackBufferStride; }
		}
		#endregion

		#region Methods WritableBitmap
		/// <summary>
		/// Specifies the area of the bitmap that changed. 
		/// </summary>
		/// <param name="dirtyRect">An Int32Rect representing the area that changed.Dimensions are in pixels.</param>
		public void AddDirtyRect(Int32Rect dirtyRect)
		{
			_hostBitmap.AddDirtyRect(dirtyRect);
		}

		/// <summary>
		/// Reserves the back buffer for updates.
		/// </summary>
		public void Lock()
		{
			_hostBitmap.Lock();
		}

		/// <summary>
		/// Attempts to lock the bitmap, waiting for no longer than the specified length of time. 
		/// </summary>
		/// <param name="timeout">A <see cref="Duration"/> that represents the length of time to wait.A value of 
		/// 0 returns immediately.A value of <see cref="Duration.Forever"/> blocks indefinitely.</param>
		/// <returns>true if the lock was acquired; otherwise, false.</returns>
		/// <remarks>When a lock is acquired, the behavior of the <see cref="TryLock"/> method is the same as the <see cref="Lock"/> method.</remarks>
		public bool TryLock(Duration timeout)
		{
			return _hostBitmap.TryLock(timeout);
		}

		/// <summary>
		/// Releases the back buffer to make it available for display. 
		/// </summary>
		/// <remarks>The Unlock method decrements the lock count.When the lock count reaches 0, 
		/// a render pass is requested if the <see cref="AddDirtyRect"/> method has been called.</remarks>
		public void Unlock()
		{
			_hostBitmap.Unlock();
		}

		/// <summary>
		/// Updates the pixels in the specified region of the bitmap. (Host only)
		/// </summary>
		/// <param name="sourceRect">The rectangle of the WriteableBitmap to update.</param>
		/// <param name="pixels">The pixel array used to update the bitmap.</param>
		/// <param name="stride">The stride of the update region in pixels.</param>
		/// <param name="offset">The input buffer offset.</param>
		public void WritePixels(Int32Rect sourceRect, Array pixels, int stride, int offset)
		{
			_hostBitmap.WritePixels(sourceRect, pixels, stride, offset);
		}

		/// <summary>
		/// Updates the pixels in the specified region of the bitmap. (Host only)
		/// </summary>
		/// <param name="sourceRect">The rectangle of the WriteableBitmap to update.</param>
		/// <param name="buffer">The input buffer used to update the bitmap.</param>
		/// <param name="bufferSize">The size of the input buffer.</param>
		/// <param name="stride">The stride of the update region in buffer.</param>
		public void WritePixels(Int32Rect sourceRect, IntPtr buffer, int bufferSize, int stride)
		{
			_hostBitmap.WritePixels(sourceRect, buffer, bufferSize, stride);
		}

		/// <summary>
		/// Updates the pixels in the specified region of the bitmap. (Host only)
		/// </summary>
		/// <param name="sourceRect">The rectangle in sourceBuffer to copy.</param>
		/// <param name="sourceBuffer">The input buffer used to update the bitmap.</param>
		/// <param name="sourceBufferStride">The stride of the input buffer, in bytes.</param>
		/// <param name="destinationX">The destination x-coordinate of the left-most pixel in the back buffer.</param>
		/// <param name="destinationY">The destination y-coordinate of the top-most pixel in the back buffer.</param>
		public void WritePixels(Int32Rect sourceRect, Array sourceBuffer, int sourceBufferStride, int destinationX, int destinationY)
		{
			_hostBitmap.WritePixels(sourceRect, sourceBuffer, sourceBufferStride, destinationX, destinationY);
		}

		/// <summary>
		/// Updates the pixels in the specified region of the bitmap. (Host only)
		/// </summary>
		/// <param name="sourceRect">The rectangle in sourceBuffer to copy.</param>
		/// <param name="sourceBuffer">The input buffer used to update the bitmap.</param>
		/// <param name="sourceBufferSize">The size of the input buffer.</param>
		/// <param name="sourceBufferStride">The stride of the input buffer, in bytes.</param>
		/// <param name="destinationX">The destination x-coordinate of the left-most pixel in the back buffer.</param>
		/// <param name="destinationY">The destination y-coordinate of the top-most pixel in the back buffer.</param>
		public void WritePixels(Int32Rect sourceRect, IntPtr sourceBuffer, int sourceBufferSize, int sourceBufferStride, int destinationX, int destinationY)
		{
			_hostBitmap.WritePixels(sourceRect, sourceBuffer, sourceBufferSize, sourceBufferStride, destinationX, destinationY);
		}
		#endregion

		#region Properties
		/// <summary>
		/// In order to show changes directly on the GUI, use the inner WritableBitmap as source.
		/// </summary>
		public WriteableBitmap InnerWritableBitmap
		{
			get { return _hostBitmap; }
		}

		/// <summary>
		/// Device pointer to allocated buffer on device
		/// </summary>
		public CUdeviceptr DevicePointer
		{
			get { return _deviceBitmap; }
		}

		/// <summary>
		/// Stride or Pitch of the device buffer. <para/>
		/// If memory is allocated pitched, it returns the real pitch as given by CUDA.<para/>
		/// If memory is allocated non-pitched, it returns (pixel size in bytes) x (width in pixels).
		/// </summary>
		public SizeT DeviceStride
		{
			get { return _deviceStride; }
		}

		/// <summary>
		/// Indicates if pixel type is compatible to pitched memory allocation.<para/>
		/// If true, device buffer was allocated using cuMemAllocPitch.
		/// </summary>
		public bool IsPitched
		{
			get { return _usePitched; }
		}

		/// <summary>
		/// Returns the number of color channels of the underlying pixel format
		/// </summary>
		public int ChannelCount
		{
			get { return _channelCount; }
		}

		/// <summary>
		/// Returns the size of one pixel for one channel in bits
		/// </summary>
		public int ChannelWidth
		{
			get { return _channelWidth; }
		}

		/// <summary>
		/// Returns the size of one pixel (all channels) in bits
		/// </summary>
		public int BitsPerPixel
		{
			get { return _bitsPerPixel; }
		}

		/// <summary>
		/// Returns the size of one pixel (all channels) in bytes
		/// </summary>
		public int BytesPerPixel
		{
			get { return _bytesPerPixel; }
		}
		#endregion

		#region Methods
		/// <summary>
		/// Update host buffer. I.e. copy from device to host.
		/// </summary>
		public void UpdateHost()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			_hostBitmap.Lock();
			if (_usePitched)
			{
				CUDAMemCpy2D copyParamters = new CUDAMemCpy2D();
				copyParamters.dstHost = _hostBitmap.BackBuffer;
				copyParamters.dstMemoryType = CUMemoryType.Host;
				copyParamters.dstPitch = _hostBitmap.BackBufferStride;
				copyParamters.srcDevice = _deviceBitmap;
				copyParamters.srcMemoryType = CUMemoryType.Device;
				copyParamters.srcPitch = _deviceStride;
				copyParamters.WidthInBytes = _hostBitmap.PixelWidth * _bytesPerPixel;
				copyParamters.Height = _hostBitmap.PixelHeight;

				res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParamters);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
			else
			{
				res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(_hostBitmap.BackBuffer, _deviceBitmap, _hostBitmap.BackBufferStride * _hostBitmap.PixelHeight);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoH", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
			_hostBitmap.AddDirtyRect(new Int32Rect(0,0,_hostBitmap.PixelWidth, _hostBitmap.PixelHeight));
			
			_hostBitmap.Unlock();
		}

		/// <summary>
		/// Update device buffer. I.e. copy from host to device.
		/// </summary>
		public void UpdateDevice()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			_hostBitmap.Lock();
			if (_usePitched)
			{
				CUDAMemCpy2D copyParamters = new CUDAMemCpy2D();
				copyParamters.srcHost = _hostBitmap.BackBuffer;
				copyParamters.srcMemoryType = CUMemoryType.Host;
				copyParamters.srcPitch = _hostBitmap.BackBufferStride;
				copyParamters.dstDevice = _deviceBitmap;
				copyParamters.dstMemoryType = CUMemoryType.Device;
				copyParamters.dstPitch = _deviceStride;
				copyParamters.WidthInBytes = _hostBitmap.PixelWidth * _bytesPerPixel;
				copyParamters.Height = _hostBitmap.PixelHeight;

				res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParamters);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
			else
			{
				res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(_deviceBitmap, _hostBitmap.BackBuffer, _hostBitmap.BackBufferStride * _hostBitmap.PixelHeight);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
				if (res != CUResult.Success) throw new CudaException(res);
			}
			_hostBitmap.Unlock();
		}

		#region CudaPitchedDeviceVariable
		/// <summary>
		/// If the pixel format can be represented by a uchar4 data type a CudaPitchedDeviceVariable is returned, else null.
		/// </summary>
		public CudaPitchedDeviceVariable<uchar4> GetPitchedDeviceVariableUchar4()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			if (!_usePitched) return null;
			if (SuggestPixelType() != typeof(uchar4)) return null;
			return new CudaPitchedDeviceVariable<uchar4>(_deviceBitmap, _hostBitmap.PixelWidth, _hostBitmap.PixelHeight, _deviceStride, false);
		}

		/// <summary>
		/// If the pixel format can be represented by a float4 data type a CudaPitchedDeviceVariable is returned, else null.
		/// </summary>
		public CudaPitchedDeviceVariable<float4> GetPitchedDeviceVariableFloat4()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			if (!_usePitched) return null;
			if (SuggestPixelType() != typeof(float4)) return null;
			return new CudaPitchedDeviceVariable<float4>(_deviceBitmap, _hostBitmap.PixelWidth, _hostBitmap.PixelHeight, _deviceStride, false);
		}

		/// <summary>
		/// If the pixel format can be represented by a float data type a CudaPitchedDeviceVariable is returned, else null.
		/// </summary>
		public CudaPitchedDeviceVariable<float> GetPitchedDeviceVariableFloat()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			if (!_usePitched) return null;
			if (SuggestPixelType() != typeof(float)) return null;
			return new CudaPitchedDeviceVariable<float>(_deviceBitmap, _hostBitmap.PixelWidth, _hostBitmap.PixelHeight, _deviceStride, false);
		}

		/// <summary>
		/// If the pixel format can be represented by a ushort4 data type a CudaPitchedDeviceVariable is returned, else null.
		/// </summary>
		public CudaPitchedDeviceVariable<ushort4> GetPitchedDeviceVariableUshort4()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			if (!_usePitched) return null;
			if (SuggestPixelType() != typeof(ushort4)) return null;
			return new CudaPitchedDeviceVariable<ushort4>(_deviceBitmap, _hostBitmap.PixelWidth, _hostBitmap.PixelHeight, _deviceStride, false);
		}

		/// <summary>
		/// If the pixel format can be represented by a uint data type a CudaPitchedDeviceVariable is returned, else null.
		/// </summary>
		public CudaPitchedDeviceVariable<uint> GetPitchedDeviceVariableUint()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			if (!_usePitched) return null;
			if (SuggestPixelType() != typeof(uint)) return null;
			return new CudaPitchedDeviceVariable<uint>(_deviceBitmap, _hostBitmap.PixelWidth, _hostBitmap.PixelHeight, _deviceStride, false);
		}

		/// <summary>
		/// Returns a CudaPitchedDeviceVariable of type T.
		/// If the device image buffer is not allocated pitched GetPitchedDeviceVariable() returns null.
		/// </summary>
		public CudaPitchedDeviceVariable<T> GetPitchedDeviceVariable<T>() where T : struct
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			if (!_usePitched) return null;
			return new CudaPitchedDeviceVariable<T>(_deviceBitmap, _hostBitmap.PixelWidth, _hostBitmap.PixelHeight, _deviceStride, false);
		}
		#endregion

		#region CudaDeviceVariable
		/// <summary>
		/// If the pixel format can be represented by a byte data type a CudaDeviceVariable is returned, else null.
		/// </summary>
		public CudaDeviceVariable<byte> GetDeviceVariableByte()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			if (_usePitched) return null;
			if (SuggestPixelType() != typeof(byte)) return null;
			return new CudaDeviceVariable<byte>(_deviceBitmap, false, _deviceStride * _hostBitmap.PixelHeight);
		}

		/// <summary>
		/// If the pixel format can be represented by a uchar3 data type a CudaDeviceVariable is returned, else null.
		/// </summary>
		public CudaDeviceVariable<uchar3> GetDeviceVariableUchar3()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			if (_usePitched) return null;
			if (SuggestPixelType() != typeof(uchar3)) return null;
			return new CudaDeviceVariable<uchar3>(_deviceBitmap, false, _deviceStride * _hostBitmap.PixelHeight);
		}

		/// <summary>
		/// If the pixel format can be represented by a ushort data type a CudaDeviceVariable is returned, else null.
		/// </summary>
		public CudaDeviceVariable<ushort> GetDeviceVariableUshort()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			if (_usePitched) return null;
			if (SuggestPixelType() != typeof(ushort)) return null;
			return new CudaDeviceVariable<ushort>(_deviceBitmap, false, _deviceStride * _hostBitmap.PixelHeight);
		}

		/// <summary>
		/// If the pixel format can be represented by a ushort3 data type a CudaDeviceVariable is returned, else null.
		/// </summary>
		public CudaDeviceVariable<ushort3> GetDeviceVariableUshort3()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			if (_usePitched) return null;
			if (SuggestPixelType() != typeof(ushort3)) return null;
			return new CudaDeviceVariable<ushort3>(_deviceBitmap, false, _deviceStride * _hostBitmap.PixelHeight);
		}

		/// <summary>
		/// Returns a CudaDeviceVariable of type T.
		/// If the device image buffer is allocated pitched GetDeviceVariable() returns null.
		/// </summary>
		public CudaDeviceVariable<T> GetDeviceVariable<T>() where T : struct
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			if (_usePitched) return null;
			return new CudaDeviceVariable<T>(_deviceBitmap, false, _deviceStride * _hostBitmap.PixelHeight);
		}
		#endregion



		#region Copy sync
		/// <summary>
		/// Copy data from device to device memory. Assuming that source has same memory layout as image data.
		/// </summary>
		/// <param name="source">Source pointer to device memory</param>
		public void CopyToDevice(CUdeviceptr source)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			if (_usePitched)
			{
				CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
				copyParams.srcDevice = source;
				copyParams.srcMemoryType = CUMemoryType.Device;
				copyParams.srcPitch = _deviceStride;
				copyParams.dstDevice = _deviceBitmap;
				copyParams.dstMemoryType = CUMemoryType.Device;
				copyParams.Height = _hostBitmap.PixelHeight;
				copyParams.WidthInBytes = _hostBitmap.PixelWidth * _bytesPerPixel;
				copyParams.dstPitch = _deviceStride;

				res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
				if (res != CUResult.Success)
					throw new CudaException(res);
			}
			else
			{
				SizeT aSizeInBytes = _deviceStride * _hostBitmap.PixelHeight;
				CUResult res;
				res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoD_v2(_deviceBitmap, source, aSizeInBytes);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoD", res));
				if (res != CUResult.Success)
					throw new CudaException(res);
			}
		}

		/// <summary>
		/// Copy data from device to device memory
		/// </summary>
		/// <param name="source">Source</param>
		public void CopyToDevice<T>(CudaDeviceVariable<T> source) where T : struct
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			if (_usePitched)
			{
				CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
				copyParams.srcDevice = source.DevicePointer;
				copyParams.srcMemoryType = CUMemoryType.Device;
				copyParams.srcPitch = _hostBitmap.PixelWidth * _bytesPerPixel;
				copyParams.dstDevice = _deviceBitmap;
				copyParams.dstMemoryType = CUMemoryType.Device;
				copyParams.Height = _hostBitmap.PixelHeight;
				copyParams.WidthInBytes = _hostBitmap.PixelWidth * _bytesPerPixel;
				copyParams.dstPitch = _deviceStride;

				res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
				if (res != CUResult.Success)
					throw new CudaException(res);
			}
			else
			{
				SizeT aSizeInBytes = _deviceStride * _hostBitmap.PixelHeight;
				CUResult res;
				res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoD_v2(_deviceBitmap, source.DevicePointer, aSizeInBytes);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyDtoD", res));
				if (res != CUResult.Success)
					throw new CudaException(res);
			}
		}

		/// <summary>
		/// Copy from device to device memory
		/// </summary>
		/// <param name="deviceSrc">Source</param>
		public void CopyToDevice<T>(CudaPitchedDeviceVariable<T> deviceSrc) where T : struct
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			
			CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
			copyParams.srcDevice = deviceSrc.DevicePointer;
			copyParams.srcMemoryType = CUMemoryType.Device;
			copyParams.srcPitch = deviceSrc.Pitch;
			copyParams.dstDevice = _deviceBitmap;
			copyParams.dstMemoryType = CUMemoryType.Device;
			copyParams.Height = deviceSrc.Height;
			copyParams.WidthInBytes = deviceSrc.WidthInBytes;
			copyParams.dstPitch = _deviceStride;

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Copy data from host to device memory
		/// </summary>
		/// <param name="source">Source pointer to host memory</param>
		/// <param name="pitch">Line pitch</param>
		public void CopyToDevice<T>(T[] source, SizeT pitch) where T : struct
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			SizeT aSizeInBytes = source.Length * _bytesPerPixel;
			GCHandle handle = GCHandle.Alloc(source, GCHandleType.Pinned);
			CUResult res;
			try
			{
				IntPtr ptr = handle.AddrOfPinnedObject();
				if (_usePitched)
				{
					CUDAMemCpy2D copyParams = new CUDAMemCpy2D();
					copyParams.srcHost = ptr;
					copyParams.srcMemoryType = CUMemoryType.Host;
					copyParams.srcPitch = pitch;
					copyParams.dstDevice = _deviceBitmap;
					copyParams.dstMemoryType = CUMemoryType.Device;
					copyParams.Height = _hostBitmap.PixelHeight;
					copyParams.WidthInBytes = _hostBitmap.PixelWidth * _bytesPerPixel;
					copyParams.dstPitch = _deviceStride;

					res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpy2D_v2(ref copyParams);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpy2D", res));
					if (res != CUResult.Success)
						throw new CudaException(res);
				}
				else
				{
					res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(_deviceBitmap, ptr, aSizeInBytes);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemcpyHtoD", res));
				}
			}
			finally
			{
				handle.Free();
			}
			if (res != CUResult.Success)
				throw new CudaException(res);
		}
		#endregion

		/// <summary>
		/// returns a type which bests fits the pixel representation in device memory.<para/>
		/// If no type can be found (for example: 1 bit BlackWhite images) null is returned.
		/// </summary>
		/// <returns></returns>
		public Type SuggestPixelType()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			if (_hostBitmap == null) return null;
			if (_hostBitmap.Format == PixelFormats.Bgr101010) return typeof(uint);
			if (_hostBitmap.Format == PixelFormats.Bgr24) return typeof(uchar3);
			if (_hostBitmap.Format == PixelFormats.Bgr32) return typeof(uchar4); //alpha is unused
			if (_hostBitmap.Format == PixelFormats.Bgr555) return typeof(ushort);
			if (_hostBitmap.Format == PixelFormats.Bgr565) return typeof(ushort);
			if (_hostBitmap.Format == PixelFormats.Bgra32) return typeof(uchar4);
			if (_hostBitmap.Format == PixelFormats.BlackWhite) return null;
			if (_hostBitmap.Format == PixelFormats.Cmyk32) return typeof(uchar4);
			if (_hostBitmap.Format == PixelFormats.Gray16) return typeof(ushort);
			if (_hostBitmap.Format == PixelFormats.Gray2) return null;
			if (_hostBitmap.Format == PixelFormats.Gray32Float) return typeof(float);
			if (_hostBitmap.Format == PixelFormats.Gray4) return null;
			if (_hostBitmap.Format == PixelFormats.Gray8) return typeof(byte);
			if (_hostBitmap.Format == PixelFormats.Indexed1) return null;
			if (_hostBitmap.Format == PixelFormats.Indexed2) return null;
			if (_hostBitmap.Format == PixelFormats.Indexed4) return null;
			if (_hostBitmap.Format == PixelFormats.Indexed8) return typeof(byte);
			if (_hostBitmap.Format == PixelFormats.Pbgra32) return typeof(uchar4);
			if (_hostBitmap.Format == PixelFormats.Prgba128Float) return typeof(float4);
			if (_hostBitmap.Format == PixelFormats.Prgba64) return typeof(ushort4); //Documentation says 32, but that doesn't sum up to 64
			if (_hostBitmap.Format == PixelFormats.Rgb128Float) return typeof(float4); //alpha is unused;
			if (_hostBitmap.Format == PixelFormats.Rgb24) return typeof(uchar3);
			if (_hostBitmap.Format == PixelFormats.Rgb48) return typeof(ushort3);
			if (_hostBitmap.Format == PixelFormats.Rgba128Float) return typeof(float4);
			if (_hostBitmap.Format == PixelFormats.Rgba64) return typeof(ushort4);
			return null;
		}

		/// <summary>
		/// Tries to find a CUArrayFormat matching the pixel format.
		/// </summary>
		/// <returns></returns>
		public CUArrayFormat SuggestCUArrayFormat()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			if (_hostBitmap == null) return 0;
			if (_hostBitmap.Format == PixelFormats.Bgr101010) return CUArrayFormat.UnsignedInt32;
			if (_hostBitmap.Format == PixelFormats.Bgr24) return CUArrayFormat.UnsignedInt8;
			if (_hostBitmap.Format == PixelFormats.Bgr32) return CUArrayFormat.UnsignedInt8; //alpha is unused
			if (_hostBitmap.Format == PixelFormats.Bgr555) return CUArrayFormat.UnsignedInt16;
			if (_hostBitmap.Format == PixelFormats.Bgr565) return CUArrayFormat.UnsignedInt16;
			if (_hostBitmap.Format == PixelFormats.Bgra32) return CUArrayFormat.UnsignedInt8;
			if (_hostBitmap.Format == PixelFormats.BlackWhite) return 0;
			if (_hostBitmap.Format == PixelFormats.Cmyk32) return CUArrayFormat.UnsignedInt8;
			if (_hostBitmap.Format == PixelFormats.Gray16) return CUArrayFormat.UnsignedInt16;
			if (_hostBitmap.Format == PixelFormats.Gray2) return 0;
			if (_hostBitmap.Format == PixelFormats.Gray32Float) return CUArrayFormat.Float;
			if (_hostBitmap.Format == PixelFormats.Gray4) return 0;
			if (_hostBitmap.Format == PixelFormats.Gray8) return CUArrayFormat.UnsignedInt8;
			if (_hostBitmap.Format == PixelFormats.Indexed1) return 0;
			if (_hostBitmap.Format == PixelFormats.Indexed2) return 0;
			if (_hostBitmap.Format == PixelFormats.Indexed4) return 0;
			if (_hostBitmap.Format == PixelFormats.Indexed8) return CUArrayFormat.UnsignedInt8;
			if (_hostBitmap.Format == PixelFormats.Pbgra32) return CUArrayFormat.UnsignedInt8;
			if (_hostBitmap.Format == PixelFormats.Prgba128Float) return CUArrayFormat.Float;
			if (_hostBitmap.Format == PixelFormats.Prgba64) return CUArrayFormat.UnsignedInt16; //Documentation says 32, but that doesn't sum up to 64
			if (_hostBitmap.Format == PixelFormats.Rgb128Float) return CUArrayFormat.Float; //alpha is unused;
			if (_hostBitmap.Format == PixelFormats.Rgb24) return CUArrayFormat.UnsignedInt8;
			if (_hostBitmap.Format == PixelFormats.Rgb48) return CUArrayFormat.UnsignedInt16;
			if (_hostBitmap.Format == PixelFormats.Rgba128Float) return CUArrayFormat.Float;
			if (_hostBitmap.Format == PixelFormats.Rgba64) return CUArrayFormat.UnsignedInt16;
			return 0;
		}

		#region Texture
		/// <summary>
		/// Creates a texture reference from the underlying device picture buffer. <para/>
		/// If the buffer was not allocated "pitched" or the pixel type cannot be determined from <see cref="PixelFormat"/> automatically, null is returned.<para/>
		/// Uses reflection to handle dynamic types.
		/// </summary>
		/// <param name="kernel"></param>
		/// <param name="texName"></param>
		/// <param name="addressMode"></param>
		/// <param name="filterMode"></param>
		/// <param name="flags"></param>
		/// <returns></returns>
		public object CreateTextureFromImage(CudaKernel kernel, string texName, CUAddressMode addressMode, CUFilterMode filterMode, CUTexRefSetFlags flags)
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			if (!_usePitched || _hostBitmap == null) return null;

			Type devVar = typeof(CudaPitchedDeviceVariable<>);
			Type devVarElem = devVar.MakeGenericType(SuggestPixelType());

			Type[] argTypesDevVar = { typeof(CUdeviceptr), typeof(SizeT), typeof(SizeT), typeof(SizeT) };
			System.Reflection.ConstructorInfo ciDevVar = devVarElem.GetConstructor(argTypesDevVar);
			object[] argsDevVar = { _deviceBitmap, (SizeT)_hostBitmap.PixelWidth, (SizeT)_hostBitmap.PixelHeight, (SizeT)_deviceStride };
			object devVarObject = ciDevVar.Invoke(argsDevVar);


			Type tex = typeof(CudaTextureLinearPitched2D<>);
			Type elem = tex.MakeGenericType(SuggestPixelType());

			Type[] argTypes = { typeof(CudaKernel), typeof(string), typeof(CUAddressMode), typeof(CUAddressMode), typeof(CUFilterMode), typeof(CUTexRefSetFlags), typeof(CUArrayFormat), devVarElem };

			System.Reflection.ConstructorInfo ci = elem.GetConstructor(argTypes);

			object[] args = { kernel, texName, addressMode, addressMode, filterMode, flags, SuggestCUArrayFormat(), devVarObject };

			return ci.Invoke(args);
		}

		/// <summary>
		/// Creates a texture reference from the underlying device picture buffer with type T.<para/>
		/// If the buffer was not allocated "pitched", null is returned.
		/// </summary>
		/// <param name="kernel"></param>
		/// <param name="texName"></param>
		/// <param name="addressMode"></param>
		/// <param name="filterMode"></param>
		/// <param name="flags"></param>
		/// <returns></returns>
		public CudaTextureLinearPitched2D<T> CreateTextureFromImage<T>(CudaKernel kernel, string texName, CUAddressMode addressMode, CUFilterMode filterMode, CUTexRefSetFlags flags) where T : struct
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			if (!_usePitched || _hostBitmap == null) return null;

			CudaPitchedDeviceVariable<T> devVar = GetPitchedDeviceVariable<T>();

			return new CudaTextureLinearPitched2D<T>(kernel, texName, addressMode, addressMode, filterMode, flags, SuggestCUArrayFormat(), devVar);
		}
		#endregion
		#endregion

	}
}
