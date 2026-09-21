# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /mnt/fast/nobackup/users/jg02228/overlap-loss-experiments/torchinductor_jg02228/6s/c6sq6dzezxmrsbuq6s3oqsf6ngugfa4riulikzkbjl4okw43ul5p.py
# Topologically Sorted Source Nodes: [norm, a], Original ATen: [aten.linalg_vector_norm, aten.div]
# Source node to ATen node mapping:
#   a => div
#   norm => pow_1, sum_1
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg2_1, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [-1]), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg2_1, %unsqueeze), kwargs = {})
triton_red_fused_div_linalg_vector_norm_0 = async_compile.triton('triton_red_fused_div_linalg_vector_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_linalg_vector_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'C3F0080BFB66D05FF6D5D3B4AF08B6B04C1D159DC9CF54D3B25BB8FC047B99A5', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_div_linalg_vector_norm_0(in_ptr0, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 * tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp5 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = libdevice.sqrt(tmp3)
        tmp7 = (tmp5 / tmp6)
        tl.store(out_ptr1 + (r0_1 + ks0*x0), tmp7, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /mnt/fast/nobackup/users/jg02228/overlap-loss-experiments/torchinductor_jg02228/ls/clstx6nvspcaat6pjphewpaywmp3s42e6caathwgza3ai45r2p3u.py
# Topologically Sorted Source Nodes: [norm_1, b], Original ATen: [aten.linalg_vector_norm, aten.div]
# Source node to ATen node mapping:
#   b => div_1
#   norm_1 => pow_3, sum_2
# Graph fragment:
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg4_1, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [-1]), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg4_1, %unsqueeze_1), kwargs = {})
triton_red_fused_div_linalg_vector_norm_1 = async_compile.triton('triton_red_fused_div_linalg_vector_norm_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_linalg_vector_norm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'C3F0080BFB66D05FF6D5D3B4AF08B6B04C1D159DC9CF54D3B25BB8FC047B99A5', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_div_linalg_vector_norm_1(in_ptr0, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 * tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp5 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = libdevice.sqrt(tmp3)
        tmp7 = (tmp5 / tmp6)
        tl.store(out_ptr1 + (r0_1 + ks0*x0), tmp7, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /mnt/fast/nobackup/users/jg02228/overlap-loss-experiments/torchinductor_jg02228/z2/cz2h6ub46xzbli5yv7dxbfia3qqfa6m6fvdue2x6rhxv2ncewcim.py
# Topologically Sorted Source Nodes: [dists], Original ATen: [aten.rsub]
# Source node to ATen node mapping:
#   dists => sub_14
# Graph fragment:
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %mm), kwargs = {})
triton_poi_fused_rsub_2 = async_compile.triton('triton_poi_fused_rsub_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_rsub_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C3F0080BFB66D05FF6D5D3B4AF08B6B04C1D159DC9CF54D3B25BB8FC047B99A5', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_rsub_2(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 1.0
    tmp2 = tmp1 - tmp0
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /mnt/fast/nobackup/users/jg02228/overlap-loss-experiments/torchinductor_jg02228/j7/cj77vtqqxv7zxqrp5aspcw7sijn34xmjh2vprrkttvewewtpqu6g.py
# Topologically Sorted Source Nodes: [gather], Original ATen: [aten.gather]
# Source node to ATen node mapping:
#   gather => gather
# Graph fragment:
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%arg7_1, 0, %expand), kwargs = {})
triton_poi_fused_gather_3 = async_compile.triton('triton_poi_fused_gather_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*u8', 'out_ptr0': '*u8', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gather_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C3F0080BFB66D05FF6D5D3B4AF08B6B04C1D159DC9CF54D3B25BB8FC047B99A5', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gather_3(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // ks0
    x0 = (xindex % ks0)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((triton_helpers.div_floor_integer(x1,  ((ks1) * ((ks1) <= (ks2)) + (ks2) * ((ks2) < (ks1)))))*((1) * ((1) >= (((ks1) * ((ks1) <= (ks2)) + (ks2) * ((ks2) < (ks1))))) + (((ks1) * ((ks1) <= (ks2)) + (ks2) * ((ks2) < (ks1)))) * ((((ks1) * ((ks1) <= (ks2)) + (ks2) * ((ks2) < (ks1)))) > (1))) + ((x1 % ((ks1) * ((ks1) <= (ks2)) + (ks2) * ((ks2) < (ks1)))))), xmask, eviction_policy='evict_last')
    tmp1 = ks2
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < ks2)) | ~(xmask), "index out of bounds: 0 <= tmp4 < ks2")
    tmp6 = tl.load(in_ptr1 + (x0 + ks0*tmp4), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = args
    args.clear()
    s55 = arg0_1
    s6 = arg1_1
    s23 = arg3_1
    s14 = arg5_1
    s87 = arg6_1
    assert_size_stride(arg2_1, (s55, s6), (s6, 1))
    assert_size_stride(arg4_1, (s23, s6), (s6, 1))
    assert_size_stride(arg7_1, (s23, s87), (s87, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((s55, s6), (s6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [norm, a], Original ATen: [aten.linalg_vector_norm, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_linalg_vector_norm_0.run(arg2_1, buf2, s6, s55, s6, stream=stream0)
        del arg2_1
        buf3 = empty_strided_cuda((s23, s6), (s6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [norm_1, b], Original ATen: [aten.linalg_vector_norm, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused_div_linalg_vector_norm_1.run(arg4_1, buf3, s6, s23, s6, stream=stream0)
        del arg4_1
        buf4 = empty_strided_cuda((s55, s23), (s23, 1), torch.float32)
        # Topologically Sorted Source Nodes: [a, matmul], Original ATen: [aten.div, aten.mm]
        extern_kernels.mm(buf2, reinterpret_tensor(buf3, (s6, s23), (1, s6), 0), out=buf4)
        del buf2
        del buf3
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [dists], Original ATen: [aten.rsub]
        triton_poi_fused_rsub_2_xnumel = s23*s55
        stream0 = get_raw_stream(0)
        triton_poi_fused_rsub_2.run(buf5, triton_poi_fused_rsub_2_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [dists, topk], Original ATen: [aten.rsub, aten.topk]
        buf6 = torch.ops.aten.topk.default(buf5, min(s14, s23), -1, False)
        del buf5
        buf7 = buf6[0]
        assert_size_stride(buf7, (s55, min(s14, s23)), (max(1, min(s14, s23)), 1), 'torch.ops.aten.topk.default')
        assert_alignment(buf7, 16, 'torch.ops.aten.topk.default')
        buf8 = buf6[1]
        assert_size_stride(buf8, (s55, min(s14, s23)), (max(1, min(s14, s23)), 1), 'torch.ops.aten.topk.default')
        assert_alignment(buf8, 16, 'torch.ops.aten.topk.default')
        del buf6
        buf9 = empty_strided_cuda((s55*min(s14, s23), s87), (s87, 1), torch.uint8)
        # Topologically Sorted Source Nodes: [gather], Original ATen: [aten.gather]
        triton_poi_fused_gather_3_xnumel = s55*s87*min(s14, s23)
        stream0 = get_raw_stream(0)
        triton_poi_fused_gather_3.run(buf8, arg7_1, buf9, s87, s14, s23, triton_poi_fused_gather_3_xnumel, stream=stream0)
        del arg7_1
        del buf8
    return (buf7, reinterpret_tensor(buf9, (s55, min(s14, s23), s87), (s87*min(s14, s23), s87 // min(s14, s23), 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 1024
    arg1_1 = 384
    arg2_1 = rand_strided((1024, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = 262144
    arg4_1 = rand_strided((262144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = 1
    arg6_1 = 256
    arg7_1 = rand_strided((262144, 256), (256, 1), device='cuda:0', dtype=torch.uint8)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
