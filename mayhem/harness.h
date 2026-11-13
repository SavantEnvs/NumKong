// Shared plumbing for the NumKong fuzz targets: capability selection and exact-size heap buffers.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numkong/numkong.h>

static nk_capability_t g_enabled = nk_cap_serial_k;

// The enabled mask is what the library itself reports for this CPU; AMX needs a per-thread opt-in.
extern "C" int LLVMFuzzerInitialize(int *, char ***) {
    if (nk_cpu_capabilities_enabled(&g_enabled) != nk_success_k) g_enabled = nk_cap_serial_k;
    (void)nk_cpu_configure_thread(g_enabled);
    return 0;
}

// Zero selects every enabled capability; otherwise a subset of them, with serial always present.
static inline nk_capability_t pick_capabilities(uint64_t mask) {
    return mask ? ((mask & g_enabled) | nk_cap_serial_k) : g_enabled;
}

struct Reader {
    uint8_t const *data;
    size_t size;
    template <typename T>
    T take() {
        T value = 0;
        size_t const bytes = size < sizeof(T) ? size : sizeof(T);
        memcpy(&value, data, bytes);
        data += bytes, size -= bytes;
        return value;
    }
};

// A heap copy of exactly `bytes` bytes, so ASan sees any read or write past the logical end.
struct Heap {
    void *pointer;
    explicit Heap(size_t bytes) : pointer(calloc(bytes ? bytes : 1, 1)) {}
    Heap(void const *source, size_t bytes) : pointer(malloc(bytes ? bytes : 1)) {
        if (bytes) memcpy(pointer, source, bytes);
    }
    ~Heap() { free(pointer); }
    Heap(Heap const &) = delete;
    Heap &operator=(Heap const &) = delete;
    Heap(Heap &&other) noexcept : pointer(other.pointer) { other.pointer = nullptr; }
    Heap &operator=(Heap &&other) noexcept {
        free(pointer);
        pointer = other.pointer, other.pointer = nullptr;
        return *this;
    }
};

static inline size_t storage_bytes(nk_dtype_t dtype, size_t dimensions) {
    return dimensions * nk_dtype_bits(dtype) / 8;
}
