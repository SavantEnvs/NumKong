#!/usr/bin/env bash
# Builds NumKong's static library twice (sanitized for the fuzz targets, normal flags for the upstream
# test suite), then links every harness against the sanitized one.
set -euo pipefail

[ -n "${SOURCE_DATE_EPOCH:-}" ] || unset SOURCE_DATE_EPOCH

: "${SANITIZER_FLAGS=-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer}"
: "${DEBUG_FLAGS:=-g -gdwarf-3}"
: "${CC:=clang}" ; : "${CXX:=clang++}" ; : "${LIB_FUZZING_ENGINE:=-fsanitize=fuzzer}"
: "${MAYHEM_JOBS:=$(nproc)}"
: "${COVERAGE_FLAGS=}"
export SANITIZER_FLAGS DEBUG_FLAGS CC CXX LIB_FUZZING_ENGINE MAYHEM_JOBS COVERAGE_FLAGS

cd "$SRC"

FMT_SRC=/opt/toolchains/deps/fmt-12.2.0/src
[ -f "$FMT_SRC/CMakeLists.txt" ] || { echo "missing pre-fetched {fmt} at $FMT_SRC" >&2; exit 1; }

TARGETS=(fuzz_dot fuzz_metric fuzz_sparse fuzz_reduce fuzz_cast)

# `-fno-sanitize=function` is the one UBSan check relaxed: the dispatcher stores every kernel as `void (*)(void)` and the
# callers cast it back to the kind's `nk_*_punned_t` signature, so each dispatched call mismatches the callee's type by design.
NO_FUNCTION_CHECK=-fno-sanitize=function

# -O1 rather than the project's -O3: the ISA-specific units (AVX-512, AMX) take minutes each at -O3 under ASan.
OPT=-O1

# 1) The upstream test suite, normal flags, no sanitizers. {fmt} comes from the image, not the network.
cmake -S . -B build-mayhem-test -G Ninja -DCMAKE_BUILD_TYPE=None \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_C_FLAGS="$OPT -g -gdwarf-3 $COVERAGE_FLAGS" -DCMAKE_CXX_FLAGS="$OPT -g -gdwarf-3 $COVERAGE_FLAGS" \
    -DCMAKE_EXE_LINKER_FLAGS="$COVERAGE_FLAGS" \
    -DNUMKONG_BUILD_SHARED=OFF -DNUMKONG_INSTALL=OFF -DNUMKONG_BUILD_TEST=ON \
    -DFETCHCONTENT_SOURCE_DIR_FMT="$FMT_SRC" -DFETCHCONTENT_FULLY_DISCONNECTED=ON
cmake --build build-mayhem-test --target numkong_cpu_test -j"$MAYHEM_JOBS"

# 2) The library under test, sanitized, with -fsanitize=fuzzer-no-link so its edges feed the fuzzer.
LIB_FLAGS="$SANITIZER_FLAGS $NO_FUNCTION_CHECK $DEBUG_FLAGS -fsanitize=fuzzer-no-link $OPT"
cmake -S . -B build-mayhem-san -G Ninja -DCMAKE_BUILD_TYPE=None \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_C_FLAGS="$LIB_FLAGS" -DCMAKE_CXX_FLAGS="$LIB_FLAGS" \
    -DNUMKONG_BUILD_SHARED=OFF -DNUMKONG_INSTALL=OFF
cmake --build build-mayhem-san --target numkong_static -j"$MAYHEM_JOBS"
LIB="$SRC/build-mayhem-san/libnumkong_static.a"

# 3) Harnesses: the libFuzzer binary and a run-once `-standalone` reproducer for each.
HARNESS_FLAGS="$SANITIZER_FLAGS $NO_FUNCTION_CHECK $DEBUG_FLAGS $OPT -I$SRC/c -I$SRC/include -I$SRC/mayhem"
$CXX $SANITIZER_FLAGS $DEBUG_FLAGS -c "$SRC/mayhem/lsan_off.cc" -o build-mayhem-san/lsan_off.o
$CC $SANITIZER_FLAGS $DEBUG_FLAGS -c "$STANDALONE_FUZZ_MAIN" -o build-mayhem-san/standalone_main.o
for t in "${TARGETS[@]}"; do
    $CXX $HARNESS_FLAGS -std=c++17 -c "$SRC/mayhem/$t.cc" -o "build-mayhem-san/$t.o"
    $CXX $SANITIZER_FLAGS $DEBUG_FLAGS "build-mayhem-san/$t.o" build-mayhem-san/lsan_off.o "$LIB" -lm \
        $LIB_FUZZING_ENGINE -o "$SRC/$t"
    $CXX $SANITIZER_FLAGS $DEBUG_FLAGS "build-mayhem-san/$t.o" build-mayhem-san/lsan_off.o \
        build-mayhem-san/standalone_main.o "$LIB" -lm -o "$SRC/$t-standalone"
done
