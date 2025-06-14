 # TypeError: bwd (): incompatible function arguments. The following argument types are supported

 ## 问题概述
 
 ```bash
   File "/sxs/zhoufei/DynaMoE/models/encoder/vmamba.py", line 81, in backward
     du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
 RuntimeError: dout must have shape (batch_size, dim, seqlen)
 TypeError: bwd(): incompatible function arguments. The following argument types are supported:
     1. (arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor, arg4: torch.Tensor, arg5: Optional[torch.Tensor], arg6: Optional[torch.Tensor], arg7: Optional[torch.Tensor], arg8: torch.Tensor, arg9: Optional[torch.Tensor], arg10: Optional[torch.Tensor], arg11: Optional[torch.Tensor], arg12: bool, arg13: bool) - List[torch.Tensor]
 ```

 ## 解决方案
 
 参考 [VMamba/kernels/selective_scan/test_selective_scan.py at main · MzeroMiko/VMamba](https://github.com/MzeroMiko/VMamba/blob/main/kernels/selective_scan/test_selective_scan.py)
 ```bash
 @staticmethod
 def backward(ctx, dout, *args):
     if not ctx.has_z:
         u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
         z = None
         out = None
     else:
         u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
     if dout.stride(-1) != 1:
         dout = dout.contiguous()
     # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
     # backward of selective_scan_cuda with the backward of chunk).
     # Here we just pass in None and dz will be allocated in the C++ code.
     if MODE in ["mamba_ssm"]:
         du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
             u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
             False # option to recompute out_z, not used here
         )
     elif MODE in ["sstest"]:
         du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
             u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
             False, ctx.backnrows  # option to recompute out_z, not used here
         )
     elif MODE in ["sscore", "ssoflex"]:
         du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
             u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.backnrows
         )
     elif MODE in ["sscorendstate"]:
         du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
             u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
         )
         dA = dA.unsqueeze(1)
         dB = dB.unsqueeze(2)
         dC = dC.unsqueeze(2)
     else:
         raise NotImplementedError
 ```
 修改你自己的 `selective_scan_cuda.bwd` 里面的参数，进行补充 `None` 和调整位置