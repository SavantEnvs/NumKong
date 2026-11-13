// Bulk dtype conversion between every scalar type, through the dispatcher.
#include "harness.h"

static nk_dtype_t const dtypes[] = {nk_f64_k,  nk_f32_k, nk_f16_k, nk_bf16_k, nk_e4m3_k, nk_e5m2_k, nk_e2m3_k,
                                    nk_e3m2_k, nk_e2m1_k, nk_i8_k, nk_u8_k,   nk_i16_k,  nk_u16_k,  nk_i32_k,
                                    nk_u32_k,  nk_i64_k,  nk_u64_k, nk_i4_k,  nk_u4_k,   nk_u1_k};
static size_t const dtype_count = sizeof(dtypes) / sizeof(dtypes[0]);

extern "C" int LLVMFuzzerTestOneInput(uint8_t const *data, size_t size) {
    Reader in {data, size};
    if (size < 11) return -1;
    nk_capability_t const capabilities = pick_capabilities(in.take<uint64_t>());
    nk_dtype_t const from = dtypes[in.take<uint8_t>() % dtype_count];
    nk_dtype_t const to = dtypes[in.take<uint8_t>() % dtype_count];

    // The count must be a multiple of the values per byte of both types (`nk_assert_dims_`), so use whole groups of 8.
    size_t const count = in.size * 8 / nk_dtype_bits(from) / 8 * 8;
    if (!count) return -1;
    Heap source(in.data, storage_bytes(from, count));
    Heap target(storage_bytes(to, count));
    (void)nk_cast_best(source.pointer, from, target.pointer, to, count, capabilities, nullptr);
    return 0;
}
