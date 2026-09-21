# AOT ID: ['3_inference']
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


# kernel path: /mnt/fast/nobackup/users/jg02228/overlap-loss-experiments/torchinductor_jg02228/2a/c2anziermxck5xeqrmslgb4ijuq5vcmq3fjdqsbmmfsu4b47xjfa.py
# Topologically Sorted Source Nodes: [dists], Original ATen: [aten._euclidean_dist]
# Source node to ATen node mapping:
#   dists => mul_16, pow_1, sum_1
# Graph fragment:
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, -2), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg4_1, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [-1], True), kwargs = {})
triton_red_fused__euclidean_dist_0 = async_compile.triton('triton_red_fused__euclidean_dist_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__euclidean_dist_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C3F0080BFB66D05FF6D5D3B4AF08B6B04C1D159DC9CF54D3B25BB8FC047B99A5', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__euclidean_dist_0(in_ptr0, out_ptr0, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0 * tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
        tmp5 = -2.0
        tmp6 = tmp0 * tmp5
        tl.store(out_ptr1 + (r0_1 + 2*x0 + ks0*x0), tmp6, r0_mask & xmask)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (2*x0 + ks0*x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /mnt/fast/nobackup/users/jg02228/overlap-loss-experiments/torchinductor_jg02228/sx/csxozkchhnadwtjr2xjx2u3dqyh2w5yfzagonfrht7xzlh3irc2e.py
# Topologically Sorted Source Nodes: [dists], Original ATen: [aten._euclidean_dist]
# Source node to ATen node mapping:
#   dists => full_default
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([%arg3_1, 1], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused__euclidean_dist_1 = async_compile.triton('triton_poi_fused__euclidean_dist_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__euclidean_dist_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'C3F0080BFB66D05FF6D5D3B4AF08B6B04C1D159DC9CF54D3B25BB8FC047B99A5', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__euclidean_dist_1(out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 1.0
    tl.store(out_ptr0 + (2*x0 + ks0*x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /mnt/fast/nobackup/users/jg02228/overlap-loss-experiments/torchinductor_jg02228/p3/cp3fynvtz6g3avsprzpujnupygbg3ocdhjq7zc26tlh6ggzoajmp.py
# Topologically Sorted Source Nodes: [dists], Original ATen: [aten._euclidean_dist]
# Source node to ATen node mapping:
#   dists => cat_1, pow_2, sum_2
# Graph fragment:
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg2_1, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_2, [-1], True), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%arg2_1, %full_default_1, %sum_2], -1), kwargs = {})
triton_red_fused__euclidean_dist_2 = async_compile.triton('triton_red_fused__euclidean_dist_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__euclidean_dist_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'C3F0080BFB66D05FF6D5D3B4AF08B6B04C1D159DC9CF54D3B25BB8FC047B99A5', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__euclidean_dist_2(in_ptr0, out_ptr0, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0 * tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
        tl.store(out_ptr1 + (r0_1 + 2*x0 + ks0*x0), tmp0, r0_mask & xmask)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (2*x0 + ks0*x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /mnt/fast/nobackup/users/jg02228/overlap-loss-experiments/torchinductor_jg02228/wr/cwroels5cnvphskagqhvdc4weewmvlk3lijlfr6cluo4rjv6n63h.py
# Topologically Sorted Source Nodes: [dists], Original ATen: [aten._euclidean_dist]
# Source node to ATen node mapping:
#   dists => full_default_1
# Graph fragment:
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([%arg0_1, 1], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused__euclidean_dist_3 = async_compile.triton('triton_poi_fused__euclidean_dist_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__euclidean_dist_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'C3F0080BFB66D05FF6D5D3B4AF08B6B04C1D159DC9CF54D3B25BB8FC047B99A5', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__euclidean_dist_3(out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 1.0
    tl.store(out_ptr0 + (2*x0 + ks0*x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /mnt/fast/nobackup/users/jg02228/overlap-loss-experiments/torchinductor_jg02228/7n/c7nzi7jgckdunoqxe33a62ihod4wmsvy4vast2jc6ng3afnu6jfs.py
# Topologically Sorted Source Nodes: [dists], Original ATen: [aten._euclidean_dist]
# Source node to ATen node mapping:
#   dists => clamp_min, sqrt
# Graph fragment:
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mm, 0), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min,), kwargs = {})
triton_poi_fused__euclidean_dist_4 = async_compile.triton('triton_poi_fused__euclidean_dist_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__euclidean_dist_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C3F0080BFB66D05FF6D5D3B4AF08B6B04C1D159DC9CF54D3B25BB8FC047B99A5', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__euclidean_dist_4(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = libdevice.sqrt(tmp2)
    tl.store(in_out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /mnt/fast/nobackup/users/jg02228/overlap-loss-experiments/torchinductor_jg02228/bt/cbtxbrlm4vjzwxq2uauzet2rel4namd63m5e7ycnso2qbfuwflkc.py
# Topologically Sorted Source Nodes: [gather], Original ATen: [aten.gather]
# Source node to ATen node mapping:
#   gather => gather
# Graph fragment:
#   %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%arg7_1, 0, %expand_2), kwargs = {})
triton_poi_fused_gather_5 = async_compile.triton('triton_poi_fused_gather_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*u8', 'out_ptr0': '*u8', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gather_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'C3F0080BFB66D05FF6D5D3B4AF08B6B04C1D159DC9CF54D3B25BB8FC047B99A5', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gather_5(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
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
    s23 = arg0_1
    s21 = arg1_1
    s55 = arg3_1
    s14 = arg5_1
    s87 = arg6_1
    assert_size_stride(arg2_1, (s23, s21), (s21, 1))
    assert_size_stride(arg4_1, (s55, s21), (s21, 1))
    assert_size_stride(arg7_1, (s23, s87), (s87, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((s55, 2 + s21), (2 + s21, 1), torch.float32)
        buf0 = reinterpret_tensor(buf3, (s55, 1), (2 + s21, 1), s21)  # alias
        buf1 = reinterpret_tensor(buf3, (s55, s21), (2 + s21, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [dists], Original ATen: [aten._euclidean_dist]
        stream0 = get_raw_stream(0)
        triton_red_fused__euclidean_dist_0.run(arg4_1, buf0, buf1, s21, s55, s21, stream=stream0)
        del arg4_1
        buf2 = reinterpret_tensor(buf3, (s55, 1), (2 + s21, 1), 1 + s21)  # alias
        # Topologically Sorted Source Nodes: [dists], Original ATen: [aten._euclidean_dist]
        stream0 = get_raw_stream(0)
        triton_poi_fused__euclidean_dist_1.run(buf2, s21, s55, stream=stream0)
        buf7 = empty_strided_cuda((s23, 2 + s21), (2 + s21, 1), torch.float32)
        buf4 = reinterpret_tensor(buf7, (s23, 1), (2 + s21, 1), 1 + s21)  # alias
        buf5 = reinterpret_tensor(buf7, (s23, s21), (2 + s21, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [dists], Original ATen: [aten._euclidean_dist]
        stream0 = get_raw_stream(0)
        triton_red_fused__euclidean_dist_2.run(arg2_1, buf4, buf5, s21, s23, s21, stream=stream0)
        del arg2_1
        del buf0
        del buf1
        del buf2
        buf6 = reinterpret_tensor(buf7, (s23, 1), (2 + s21, 1), s21)  # alias
        # Topologically Sorted Source Nodes: [dists], Original ATen: [aten._euclidean_dist]
        stream0 = get_raw_stream(0)
        triton_poi_fused__euclidean_dist_3.run(buf6, s21, s23, stream=stream0)
        del buf4
        del buf5
        del buf6
        buf8 = empty_strided_cuda((s55, s23), (s23, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dists], Original ATen: [aten._euclidean_dist]
        extern_kernels.mm(buf3, reinterpret_tensor(buf7, (2 + s21, s23), (1, 2 + s21), 0), out=buf8)
        del buf3
        del buf7
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [dists], Original ATen: [aten._euclidean_dist]
        triton_poi_fused__euclidean_dist_4_xnumel = s23*s55
        stream0 = get_raw_stream(0)
        triton_poi_fused__euclidean_dist_4.run(buf9, triton_poi_fused__euclidean_dist_4_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [dists, topk], Original ATen: [aten._euclidean_dist, aten.topk]
        buf10 = torch.ops.aten.topk.default(buf9, min(s14, s23), -1, False)
        del buf9
        buf11 = buf10[0]
        assert_size_stride(buf11, (s55, min(s14, s23)), (max(1, min(s14, s23)), 1), 'torch.ops.aten.topk.default')
        assert_alignment(buf11, 16, 'torch.ops.aten.topk.default')
        buf12 = buf10[1]
        assert_size_stride(buf12, (s55, min(s14, s23)), (max(1, min(s14, s23)), 1), 'torch.ops.aten.topk.default')
        assert_alignment(buf12, 16, 'torch.ops.aten.topk.default')
        del buf10
        buf13 = empty_strided_cuda((s55*min(s14, s23), s87), (s87, 1), torch.uint8)
        # Topologically Sorted Source Nodes: [gather], Original ATen: [aten.gather]
        triton_poi_fused_gather_5_xnumel = s55*s87*min(s14, s23)
        stream0 = get_raw_stream(0)
        triton_poi_fused_gather_5.run(buf12, arg7_1, buf13, s87, s14, s23, triton_poi_fused_gather_5_xnumel, stream=stream0)
        del arg7_1
        del buf12
    return (buf11, reinterpret_tensor(buf13, (s55, min(s14, s23), s87), (s87*min(s14, s23), s87, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 262144
    arg1_1 = 384
    arg2_1 = rand_strided((262144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = 1024
    arg4_1 = rand_strided((1024, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = 3
    arg6_1 = 256
    arg7_1 = rand_strided((262144, 256), (256, 1), device='cuda:0', dtype=torch.uint8)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
