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
    /// Measures via CUDA events the timespan between Start() and Stop() calls.
    /// </summary>
    [Obsolete("CUDA 4.0 API calls are now thread safe. Dispatched... wrapper classes are of no use now.")]
    public class DispatchedCudaStopWatch : DispatchedCudaBaseClass, IDisposable
    {
        CUevent _start, _stop;
        CUstream _stream;
        CUResult res;
        bool disposed;

        /// <summary>
        /// 
        /// </summary>
        public DispatchedCudaStopWatch(System.Windows.Threading.Dispatcher dispatcher) : base(dispatcher)
        {
            _start = new CUevent();
            _stop = new CUevent();
            _stream = new CUstream();

            object[] paramEventCreateStart = { _start, CUEventFlags.Default };
            object[] paramEventCreateStop = { _stop, CUEventFlags.Default };

            res = (CUResult)_dispatcher.Invoke(new cuEventCreate(DriverAPINativeMethods.Events.cuEventCreate), paramEventCreateStart);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuEventCreate", res));
            //res = DriverAPINativeMethods.Events.cuEventCreate(ref _start, CUEventFlags.Default);
            if (res != CUResult.Success) throw new CudaException(res);
            _start = (CUevent)paramEventCreateStart[0];

            res = (CUResult)_dispatcher.Invoke(new cuEventCreate(DriverAPINativeMethods.Events.cuEventCreate), paramEventCreateStop);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuEventCreate", res));
            //res = DriverAPINativeMethods.Events.cuEventCreate(ref _stop, CUEventFlags.Default);
            if (res != CUResult.Success) throw new CudaException(res);
            _stop = (CUevent)paramEventCreateStop[0];
        }

        /// <summary>
        /// 
        /// </summary>
        public DispatchedCudaStopWatch(CUEventFlags flags, System.Windows.Threading.Dispatcher dispatcher)
            : base(dispatcher)
        {
            _start = new CUevent();
            _stop = new CUevent();
            _stream = new CUstream();

            object[] paramEventCreateStart = { _start, flags };
            object[] paramEventCreateStop = { _stop, flags };

            res = (CUResult)_dispatcher.Invoke(new cuEventCreate(DriverAPINativeMethods.Events.cuEventCreate), paramEventCreateStart);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuEventCreate", res));
            //res = DriverAPINativeMethods.Events.cuEventCreate(ref _start, CUEventFlags.Default);
            if (res != CUResult.Success) throw new CudaException(res);
            _start = (CUevent)paramEventCreateStart[0];

            res = (CUResult)_dispatcher.Invoke(new cuEventCreate(DriverAPINativeMethods.Events.cuEventCreate), paramEventCreateStop);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuEventCreate", res));
            //res = DriverAPINativeMethods.Events.cuEventCreate(ref _stop, CUEventFlags.Default);
            if (res != CUResult.Success) throw new CudaException(res);
            _stop = (CUevent)paramEventCreateStop[0];

        }        
        
        /// <summary>
        /// For dispose
        /// </summary>
        ~DispatchedCudaStopWatch()
        {
            Dispose (false);            
        }

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
               if (_dispatcher != null)
               {
                   _dispatcher.Invoke(new cuEventDestroy(DriverAPINativeMethods.Events.cuEventDestroy), _start);
                   Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuEventDestroy", res));
                   _dispatcher.Invoke(new cuEventDestroy(DriverAPINativeMethods.Events.cuEventDestroy), _stop);
                   Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuEventDestroy", res));
               }
               disposed = true;
           }
        }

        /// <summary>
        /// Start measurement
        /// </summary>
        public void Start()
        {
            res = (CUResult)_dispatcher.Invoke(new cuEventRecord(DriverAPINativeMethods.Events.cuEventRecord), _start, _stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuEventRecord", res));
            //res = DriverAPINativeMethods.Events.cuEventRecord(_start, _stream);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Stop measurement
        /// </summary>
        public void Stop()
        {
            res = (CUResult)_dispatcher.Invoke(new cuEventRecord(DriverAPINativeMethods.Events.cuEventRecord), _stop, _stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuEventRecord", res));
            //res = DriverAPINativeMethods.Events.cuEventRecord(_stop, _stream);
            if (res != CUResult.Success) throw new CudaException(res);

            res = (CUResult)_dispatcher.Invoke(new cuEventSynchronize(DriverAPINativeMethods.Events.cuEventSynchronize), _stop);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuEventSynchronize", res));
            res = DriverAPINativeMethods.Events.cuEventSynchronize(_stop);
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Get elapsed time
        /// </summary>
        /// <returns>Elapsed time in ms</returns>
        public float GetElapsedTime()
        {

            float ms = 0;
            object[] paramEventGetElapsedTime = { ms, _start, _stop };
            res = (CUResult)_dispatcher.Invoke(new cuEventElapsedTime(DriverAPINativeMethods.Events.cuEventElapsedTime), paramEventGetElapsedTime);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuEventElapsedTime", res));
            //res = DriverAPINativeMethods.Events.cuEventElapsedTime(ref ms, _start, _stop);
            if (res != CUResult.Success) throw new CudaException(res);
            ms = (float)paramEventGetElapsedTime[0];
            return ms;
        }
    }
}
