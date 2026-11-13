// Sorted-set intersection and weighted sparse dot products over u16/u32/u64 indices.
#include <limits>

#include "harness.h"

namespace {

// `sparse.h` requires sorted, strictly ascending indices. The input is read as gaps (+1) off a start index,
// and an input whose indices would wrap is not a valid array, so it is rejected.
template <typename index_t>
bool fill_indices(Reader &in, size_t length, index_t start, index_t *out) {
    index_t current = start;
    for (size_t i = 0; i != length; ++i) {
        if (i) {
            index_t const gap = static_cast<index_t>(in.take<uint8_t>()) + static_cast<index_t>(1);
            if (current > std::numeric_limits<index_t>::max() - gap) return false;
            current = static_cast<index_t>(current + gap);
        }
        out[i] = current;
    }
    return true;
}

struct Arrays {
    size_t a_length, b_length;
};

template <typename index_t>
bool split_arrays(Reader &in, size_t weight_bytes, Arrays &lengths, Heap &a, Heap &b, bool &ok) {
    index_t const base = in.take<index_t>();
    index_t const a_offset = static_cast<index_t>(in.take<uint8_t>());
    index_t const b_offset = static_cast<index_t>(in.take<uint8_t>());
    if (base > std::numeric_limits<index_t>::max() - 255) return ok = false;
    uint8_t const split = in.take<uint8_t>();
    size_t const elements = in.size / (1 + weight_bytes);
    lengths.a_length = elements * split / 255;
    lengths.b_length = elements - lengths.a_length;
    a = Heap(lengths.a_length * sizeof(index_t));
    b = Heap(lengths.b_length * sizeof(index_t));
    ok = fill_indices<index_t>(in, lengths.a_length, static_cast<index_t>(base + a_offset), static_cast<index_t *>(a.pointer)) &&
         fill_indices<index_t>(in, lengths.b_length, static_cast<index_t>(base + b_offset), static_cast<index_t *>(b.pointer));
    return ok;
}

template <typename index_t, typename fn_t>
int run_intersect(Reader &in, nk_capability_t capabilities, bool count_only, fn_t kernel) {
    Arrays lengths {};
    Heap a(0), b(0);
    bool ok = false;
    if (!split_arrays<index_t>(in, 0, lengths, a, b, ok)) return -1;
    size_t const capacity = lengths.a_length < lengths.b_length ? lengths.a_length : lengths.b_length;
    Heap out(capacity * sizeof(index_t));
    nk_size_t count = 0;
    (void)kernel(static_cast<index_t const *>(a.pointer), static_cast<index_t const *>(b.pointer), lengths.a_length,
                 lengths.b_length, count_only ? nullptr : static_cast<index_t *>(out.pointer), &count, capabilities,
                 nullptr);
    return 0;
}

template <typename index_t, typename weight_t, typename product_t, typename fn_t>
int run_dot(Reader &in, nk_capability_t capabilities, fn_t kernel) {
    Arrays lengths {};
    Heap a(0), b(0);
    bool ok = false;
    if (!split_arrays<index_t>(in, sizeof(weight_t), lengths, a, b, ok)) return -1;
    size_t const a_bytes = lengths.a_length * sizeof(weight_t), b_bytes = lengths.b_length * sizeof(weight_t);
    if (in.size < a_bytes + b_bytes) return -1;
    Heap a_weights(in.data, a_bytes), b_weights(in.data + a_bytes, b_bytes);
    product_t product = 0;
    (void)kernel(static_cast<index_t const *>(a.pointer), static_cast<index_t const *>(b.pointer),
                 static_cast<weight_t const *>(a_weights.pointer), static_cast<weight_t const *>(b_weights.pointer),
                 lengths.a_length, lengths.b_length, &product, capabilities, nullptr);
    return 0;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(uint8_t const *data, size_t size) {
    Reader in {data, size};
    if (size < 16) return -1;
    nk_capability_t const capabilities = pick_capabilities(in.take<uint64_t>());
    uint8_t const selector = in.take<uint8_t>();
    bool const count_only = in.take<uint8_t>() & 1;
    switch (selector % 5) {
    case 0: return run_intersect<nk_u16_t>(in, capabilities, count_only, nk_sparse_intersect_u16_best);
    case 1: return run_intersect<nk_u32_t>(in, capabilities, count_only, nk_sparse_intersect_u32_best);
    case 2: return run_intersect<nk_u64_t>(in, capabilities, count_only, nk_sparse_intersect_u64_best);
    case 3: return run_dot<nk_u16_t, nk_bf16_t, nk_f32_t>(in, capabilities, nk_sparse_dot_u16bf16_best);
    default: return run_dot<nk_u32_t, nk_f32_t, nk_f64_t>(in, capabilities, nk_sparse_dot_u32f32_best);
    }
}
