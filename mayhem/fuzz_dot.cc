// Dense inner products (and the conjugated variant for complex types) through the runtime dispatcher.
#include "harness.h"

static nk_dtype_t const dtypes[] = {nk_f64_k,  nk_f32_k,   nk_f16_k,  nk_bf16_k,  nk_e4m3_k, nk_e5m2_k,
                                    nk_e2m3_k, nk_e3m2_k,  nk_e2m1_k, nk_i8_k,    nk_u8_k,   nk_i4_k,
                                    nk_u4_k,   nk_u1_k,    nk_f64c_k, nk_f32c_k,  nk_f16c_k, nk_bf16c_k};

extern "C" int LLVMFuzzerTestOneInput(uint8_t const *data, size_t size) {
    Reader in {data, size};
    if (size < 10) return -1;
    nk_capability_t const capabilities = pick_capabilities(in.take<uint64_t>());
    uint8_t const kind_selector = in.take<uint8_t>();
    nk_dtype_t const dtype = dtypes[in.take<uint8_t>() % (sizeof(dtypes) / sizeof(dtypes[0]))];
    bool const complex_type = nk_dtype_bits(dtype) >= 32 && (dtype == nk_f64c_k || dtype == nk_f32c_k ||
                                                              dtype == nk_f16c_k || dtype == nk_bf16c_k);
    nk_kernel_kind_t const kind = (complex_type && (kind_selector & 1)) ? nk_kernel_vdot_k : nk_kernel_dot_k;

    // Dimensions count values (complex pairs for complex types, bits for u1); `a` and `b` get equal halves.
    size_t const half = in.size / 2;
    size_t const dimensions = half * 8 / nk_dtype_bits(dtype);
    if (!dimensions) return -1;
    size_t const bytes = storage_bytes(dtype, dimensions);
    Heap a(in.data, bytes), b(in.data + half, bytes);
    Heap result(nk_dtype_bits(nk_kernel_output_dtype(kind, dtype)) / 8);

    nk_kernel_punned_t kernel = nullptr;
    nk_capability_t chosen = 0;
    if (nk_find_kernel_punned(kind, dtype, capabilities, &kernel, &chosen) != nk_success_k || !kernel) return 0;
    (void)reinterpret_cast<nk_metric_dense_punned_t>(kernel)(a.pointer, b.pointer, dimensions, result.pointer, nullptr);
    return 0;
}
