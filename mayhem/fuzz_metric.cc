// Dense distances (angular, euclidean, squared euclidean, hamming, jaccard, KL and JS divergence) through the dispatcher.
#include "harness.h"

struct Family {
    nk_kernel_kind_t kind;
    nk_dtype_t dtypes[16];
    size_t count;
};

static Family const families[] = {
    {nk_kernel_angular_k,
     {nk_f64_k, nk_f32_k, nk_f16_k, nk_bf16_k, nk_e4m3_k, nk_e5m2_k, nk_e2m3_k, nk_e3m2_k, nk_e2m1_k, nk_i8_k, nk_u8_k,
      nk_i4_k, nk_u4_k},
     13},
    {nk_kernel_euclidean_k,
     {nk_f64_k, nk_f32_k, nk_f16_k, nk_bf16_k, nk_e4m3_k, nk_e5m2_k, nk_e2m3_k, nk_e3m2_k, nk_e2m1_k, nk_i8_k, nk_u8_k,
      nk_i4_k, nk_u4_k},
     13},
    {nk_kernel_sqeuclidean_k,
     {nk_f64_k, nk_f32_k, nk_f16_k, nk_bf16_k, nk_e4m3_k, nk_e5m2_k, nk_e2m3_k, nk_e3m2_k, nk_e2m1_k, nk_i8_k, nk_u8_k,
      nk_i4_k, nk_u4_k},
     13},
    {nk_kernel_hamming_k, {nk_u1_k, nk_u8_k}, 2},
    {nk_kernel_jaccard_k, {nk_u1_k, nk_u8_k}, 2},
    {nk_kernel_kld_k, {nk_f64_k, nk_f32_k, nk_f16_k, nk_bf16_k}, 4},
    {nk_kernel_jsd_k, {nk_f64_k, nk_f32_k, nk_f16_k, nk_bf16_k}, 4},
};

extern "C" int LLVMFuzzerTestOneInput(uint8_t const *data, size_t size) {
    Reader in {data, size};
    if (size < 10) return -1;
    nk_capability_t const capabilities = pick_capabilities(in.take<uint64_t>());
    Family const &family = families[in.take<uint8_t>() % (sizeof(families) / sizeof(families[0]))];
    nk_dtype_t const dtype = family.dtypes[in.take<uint8_t>() % family.count];

    size_t const half = in.size / 2;
    size_t const dimensions = half * 8 / nk_dtype_bits(dtype);
    if (!dimensions) return -1;
    size_t const bytes = storage_bytes(dtype, dimensions);
    Heap a(in.data, bytes), b(in.data + half, bytes);
    nk_dtype_t const output = nk_kernel_output_dtype(family.kind, dtype);
    Heap result(output == nk_dtype_unknown_k ? 16 : nk_dtype_bits(output) / 8);

    nk_kernel_punned_t kernel = nullptr;
    nk_capability_t chosen = 0;
    if (nk_find_kernel_punned(family.kind, dtype, capabilities, &kernel, &chosen) != nk_success_k || !kernel) return 0;
    (void)reinterpret_cast<nk_metric_dense_punned_t>(kernel)(a.pointer, b.pointer, dimensions, result.pointer, nullptr);
    return 0;
}
