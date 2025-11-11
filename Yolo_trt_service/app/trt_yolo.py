import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

class TrtRunner:
    """
    Wraps a TensorRT engine for single-input, single-output YOLO inference.
    Expects engine built for (1, 3, 640, 640) FP16 or FP32 input.
    """
    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()

        self.num_bindings = self.engine.num_bindings
        self.bindings = [None] * self.num_bindings
        self.dev_buffers = []
        self.host_output = None
        self.stream = cuda.Stream()

        for i in range(self.num_bindings):
            shape = self.ctx.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            dptr = cuda.mem_alloc(nbytes)
            self.bindings[i] = int(dptr)
            self.dev_buffers.append((dptr, nbytes, dtype, shape))

            if self.engine.binding_is_input(i):
                self.in_idx = i
                self.in_shape = shape
                self.in_dtype = dtype
            else:
                self.out_idx = i
                self.out_shape = shape
                self.out_dtype = dtype
                self.host_output = np.empty(shape, dtype=dtype)

    def infer(self, nchw_fp32: np.ndarray) -> np.ndarray:
        """
        nchw_fp32: np.ndarray (1, 3, H, W), float32 normalized 0..1
        Returns: raw output tensor from the engine.
        """
        d_in, _, _, _ = self.dev_buffers[self.in_idx]
        d_out, _, _, _ = self.dev_buffers[self.out_idx]

        cuda.memcpy_htod_async(d_in, nchw_fp32, self.stream)
        self.ctx.execute_async_v2(self.bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(self.host_output, d_out, self.stream)
        self.stream.synchronize()

        return self.host_output
