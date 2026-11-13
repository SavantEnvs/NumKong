// Strided sum / sum-of-squares and min / max with argmin / argmax reductions through the dispatcher.
#include "harness.h"

static nk_dtype_t const dtypes[] = {nk_f64_k, nk_f32_k, nk_f16_k, nk_bf16_k, nk_e4m3_k, nk_e5m2_k, nk_i8_k,
                                    nk_u8_k,  nk_i16_k, nk_u16_k, nk_i32_k,  nk_u32_k,  nk_i64_k,  nk_u64_k};

extern "C" int LLVMFuzzerTestOneInput(uint8_t const *data, size_t size) {
    Reader in {data, size};
    if (size < 11) return -1;
    nk_capability_t const capabilities = pick_capabilities(in.take<uint64_t>());
    uint8_t const selector = in.take<uint8_t>();
    nk_dtype_t const dtype = dtypes[in.take<uint8_t>() % (sizeof(dtypes) / sizeof(dtypes[0]))];
    uint8_t const stride_selector = in.take<uint8_t>();

    // The stride is in bytes; the buffer ends exactly at the last element the kernel may touch.
    size_t const element = nk_dtype_bits(dtype) / 8;
    size_t const stride = element * (1 + stride_selector % 4);
    size_t const count = in.size >= element ? (in.size - element) / stride + 1 : 0;
    if (!count) return -1;
    size_t const bytes = (count - 1) * stride + element;
    Heap values(in.data, bytes);

    bool const moments = selector & 1;
    nk_kernel_punned_t kernel = nullptr;
    nk_capability_t chosen = 0;
    if (nk_find_kernel_punned(moments ? nk_kernel_reduce_moments_k : nk_kernel_reduce_minmax_k, dtype, capabilities,
                              &kernel, &chosen) != nk_success_k ||
        !kernel)
        return 0;
    if (moments) {
        Heap sum(8), sumsq(8);
        (void)reinterpret_cast<nk_reduce_moments_punned_t>(kernel)(values.pointer, count, stride, sum.pointer,
                                                                   sumsq.pointer, nullptr);
    } else {
        Heap min_value(element), max_value(element);
        nk_size_t min_index = 0, max_index = 0;
        (void)reinterpret_cast<nk_reduce_minmax_punned_t>(kernel)(values.pointer, count, stride, min_value.pointer,
                                                                  &min_index, max_value.pointer, &max_index, nullptr);
    }
    return 0;
}
