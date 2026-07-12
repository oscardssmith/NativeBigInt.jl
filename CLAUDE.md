# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

NativeBigInt.jl is a pure-Julia arbitrary-precision integer type (`NBig`) aiming
for GMP-competitive performance in the ~100–4000 bit regime. It reimplements the
mpn/mpz split of GMP in Julia and depends only on SIMD.jl (plus an optional
Random extension). Requires a recent Julia — it uses `Memory{UInt64}`.

## Commands

```bash
# Run the full test suite (activates test deps: Test, Random)
julia --project -e 'using Pkg; Pkg.test()'

# Run a single test file — load the package, then include the file.
# (test/runtests.jl just includes the per-area files under one @testset.)
julia --project -e 'using NativeBigInt, Test, Random; include("test/test_kernels.jl")'

# Benchmarks (not part of CI; each script is standalone)
julia --project bench/bench_highlevel.jl   # NBig vs Base.BigInt, end-to-end
julia --project bench/bench_kernels.jl      # kernels vs GMP __gmpn_* directly
```

There is no build/lint step — it's a plain Julia package.

## Architecture

Three layers mirror GMP's mpn/mpz split. The module include order in
`src/NativeBigInt.jl` reflects the dependency direction (kernels → algorithms →
NBig).

**Kernels (`src/kernels/*.jl`) — the mpn layer.** Sign-free primitives operating
on raw `Memory{Limb}` (`Limb == UInt64`) buffers with explicit offsets and
lengths. `addsub.jl` (`add_n!`/`sub_n!`), `mul.jl` (`mul_1!`/`addmul_1!`,
`mul_2!`/`addmul_2!`, `mul_basecase!`, squaring), `shift.jl`, `div.jl`
(reciprocal-based `divrem_1!` via Möller–Granlund normalized inverse, `div_3by2`
quotient estimation, radix-β² `divrem_bc!` Knuth Algorithm D). Kernels are
written so LLVM emits `adc`/`mulx` chains, with SIMD.jl fast paths (`V8` =
`SIMD.Vec{8,Limb}`, the shared width) and scalar cold paths for carry-chaining
edge cases. Performance here is the whole point — changes should be checked
against `bench/` and, where relevant, the generated asm (`bench/asm_dump.jl`).

**Algorithms (`src/mul.jl`, `src/algorithms.jl`) — mpn-level composite ops.**
`mul.jl` holds the multiplication chain and its dispatch thresholds:
subtractive Karatsuba (threshold ~29 limbs, benchmark-tuned via
`bench/bench_kar_thr.jl`) with an unbalanced-operand path, and the
`mul!`/`sqr!` entry points that hand off to the two-prime CRT fp NTT
(`src/fpntt.jl`, FLINT-fft_small-style Float64 engine over
p₁ = 2^49 − 2^33 + 1 and p₂ = 255·2^41 + 1, Garner recombination in the
unpack) at ~336 balanced limbs (~400 for squaring).  The single-prime
variant (`mul_fpntt!`) is kept unwired as a test cross-check; Toom-3 and
the integer Goldilocks NTT (`src/ntt.jl`) were deleted once the fp engine
beat them everywhere (git history has them if ever needed).
`algorithms.jl` has multi-limb `divrem!` (Knuth Algorithm D over
`divrem_bc!`), Karatsuba sqrt, powermod, and radix conversion for
`string`/`parse`. `montgomery.jl` and `gcd.jl` (Lehmer gcd / extended gcd,
Knuth TAOCP §4.5.2 Algorithm L) also build on the kernels and on
`mul!`/`divrem!`.

**`NBig` (`src/nbig.jl`) — the mpz layer.** The public sign-magnitude value type:

```julia
struct NBig <: Signed
    signlen::Int         # sign(x) * limb count; 0 ⟺ x == 0
    limbs::Memory{Limb}  # little-endian, normalized (top limb ≠ 0), may be over-allocated
end
```

This is where signs, normalization, and the Base interface live: comparison,
`+`/`-`/`*`, `divrem`/`div`/`rem`/`mod`/`fld`/`cld`, shifts, two's-complement
bitwise ops, `^`, base 2–36 string conversion, and conversions to/from
`Int64`/`UInt64`/`Int128`/`BigInt`/`Float64`.

**Random extension (`ext/NativeBigIntRandomExt.jl`).** Weak-dependency extension
loaded when `Random` is available; provides `rand` over `NBig` ranges.

## Conventions and invariants

- `signlen` encodes both sign and length; `0` is the unique zero (`limbs` is
  `EMPTY_LIMBS`). Magnitude length is `nlimbs(x) = abs(x.signlen)`.
- Limb buffers are little-endian and **top-limb-normalized** (no leading zero
  limbs) at the NBig boundary; use `normlen`/`nbig_from_limbs` when constructing.
  Kernel buffers may be over-allocated — always pass explicit lengths/offsets.
- Kernels take `(buffer, offset, ..., n)` and are `@inbounds`; the caller owns
  bounds and aliasing correctness.

## Testing structure

Tests are split by layer and by strategy. `test_kernels.jl`, `test_algorithms.jl`,
`test_nbig.jl` are per-layer unit tests. `test_differential.jl` and
`test_mixed.jl` cross-check results against `Base.BigInt` over randomized inputs —
this is the primary correctness net, so new operations should be added to the
differential comparison against `BigInt`.
