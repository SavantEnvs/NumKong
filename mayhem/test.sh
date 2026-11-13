#!/usr/bin/env bash
# Runs upstream's own precision suite (built by build.sh) on a bounded subset and asserts its summary line and known rows.
set -uo pipefail
[ -n "${SOURCE_DATE_EPOCH:-}" ] || unset SOURCE_DATE_EPOCH
cd "$SRC"

emit_ctrf() {
  local tool="$1" passed="$2" failed="$3" skipped="${4:-0}" pending="${5:-0}" other="${6:-0}"
  local tests=$(( passed + failed + skipped + pending + other ))
  cat > "${CTRF_REPORT:-$SRC/ctrf-report.json}" <<JSON
{
  "results": {
    "tool": { "name": "$tool" },
    "summary": {
      "tests": $tests,
      "passed": $passed,
      "failed": $failed,
      "pending": $pending,
      "skipped": $skipped,
      "other": $other
    }
  }
}
JSON
  printf 'CTRF {"results":{"tool":{"name":"%s"},"summary":{"tests":%d,"passed":%d,"failed":%d,"pending":%d,"skipped":%d,"other":%d}}}\n' \
    "$tool" "$tests" "$passed" "$failed" "$pending" "$skipped" "$other"
  [ "$failed" -eq 0 ]
}

RUNNER="$SRC/build-mayhem-test/numkong_cpu_test"
[ -x "$RUNNER" ] || { echo "missing $RUNNER (mayhem/build.sh builds it)" >&2; emit_ctrf numkong_cpu_test 0 1; exit $?; }

# The kernels' accuracy tables for dense dot, distances, divergences, sums and the dispatcher lookups, at 10 ms per kernel.
OUT="$(mktemp)"
NUMKONG_TIME_LIMIT=10ms \
NUMKONG_FILTER='^(vector_basics|vector_reductions|best_cast|find_kernel|device_capabilities|dot_|vdot_|sqeuclidean_|angular_|euclidean_|kld_|jsd_|each_sum|hamming|jaccard|sparse|intersect)' \
  "$RUNNER" > "$OUT" 2>&1
STATUS=$?
tail -n 3 "$OUT"

# A result row is `<kernel> <numbers...> <exact>`; `exact` (last column) counts inputs the kernel got bit-exact.
ROWS="$(grep -cE '^[a-z0-9_]+ +[-0-9.e+]+( +[-0-9.e+]+)+$' "$OUT")"
FAILED="$(sed -n 's/^\([0-9][0-9]*\) kernel(s) failed accuracy checks\./\1/p' "$OUT" | tail -n 1)"
FAILED="${FAILED:-0}"
PASSED_LINE="$(grep -c '^All tests passed\.$' "$OUT")"
exact_of() { awk -v k="$1" '$1==k {print $NF}' "$OUT" | head -n 1; }

CHECK_FAILS=0
check() { if [ "${2:-0}" -gt 0 ] 2>/dev/null; then echo "ok: $1 = $2"; else echo "FAIL: $1 = ${2:-missing}" >&2; CHECK_FAILS=$((CHECK_FAILS+1)); fi; }
# Known answers: f64 and i8 dot products must be bit-exact against the reference on some inputs.
check "dot_f64_serial exact inputs" "$(exact_of dot_f64_serial)"
check "dot_i8_serial exact inputs" "$(exact_of dot_i8_serial)"
check "sqeuclidean_f64_serial exact inputs" "$(exact_of sqeuclidean_f64_serial)"
check "result rows" "$ROWS"

FAIL_TOTAL=$(( FAILED + CHECK_FAILS ))
[ "$STATUS" -eq 0 ] || FAIL_TOTAL=$(( FAIL_TOTAL + 1 ))
[ "$PASSED_LINE" -ge 1 ] || FAIL_TOTAL=$(( FAIL_TOTAL + 1 ))
emit_ctrf numkong_cpu_test "$ROWS" "$FAIL_TOTAL"
