
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
