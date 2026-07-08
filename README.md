# NativeBigInt.jl

Pure-Julia arbitrary-precision integer type (`NBig`) targeting GMP-competitive
performance in the ~100–4000 bit regime. Requires a recent Julia (uses
`Memory{UInt64}`).

## Algorithms

Three layers, mirroring GMP's mpn/mpz split:

- **Kernels (`src/kernels.jl`):** sign-free limb-vector primitives —
  `add_n!`/`sub_n!`, `mul_1!`/`addmul_1!`/`submul_1!`, `mul_2!`/`addmul_2!`,
  `mul_basecase!`, `lshift!`/`rshift!`, `cmp_limbs`. Division uses a
  reciprocal-based `divrem_1!` (Möller–Granlund 2/1 normalized inverse, with
  a fused-shift path for unnormalized divisors), `div_3by2` for quotient
  estimation, and `divrem_bc!` — a radix-β² Knuth Algorithm D that produces
  two quotient limbs per pass via chained `div_3by2` estimates and a SIMD
  `submul_2!`, with a small-quotient subtraction fast path for near-equal
  bit lengths. Kernels are written so LLVM emits `adc`/`mulx` chains, with
  SIMD.jl fast paths and scalar cold paths for carry-chaining edge cases.
- **Algorithms (`src/algorithms.jl`):** subtractive Karatsuba multiplication
  (threshold ~29 limbs, benchmark-tuned) with a general unbalanced-operand
  path; multi-limb `divrem!` (Knuth Algorithm D basecase over `divrem_bc!`);
  power by repeated squaring; radix conversion for `string`/`parse`
  (per-limb `divrem_1!` for small values, divide-and-conquer for large).
- **`NBig` (`src/nbig.jl`):** sign-magnitude value type
  (`signlen = sign * limb count`, little-endian normalized `Memory{UInt64}`),
  covering comparison, `+`/`-`/`*`, `divrem`/`div`/`rem`/`mod`/`fld`/`cld`,
  shifts, two's-complement bitwise ops, `^`, base 2–36 string conversion,
  and conversions to/from `Int64`/`UInt64`/`Int128`/`BigInt`/`Float64`.

## Benchmarks

`bench/bench_highlevel.jl` compares `NBig` against `Base.BigInt` end-to-end
(construction not included) across the target bit-size regime. Ratio is
NBig time / BigInt time — lower is better; ≤1.0 means NBig is faster.

| op       | 128 bits | 256 bits | 512 bits | 1024 bits | 2048 bits | 4096 bits |
|----------|----------|----------|----------|-----------|-----------|-----------|
| `+`      | 0.50     | 0.49     | 0.55     | 0.59      | 0.66      | 0.83      |
| `-`      | 0.52     | 0.50     | 0.51     | 0.55      | 0.73      | 0.76      |
| `*`      | 0.41     | 0.40     | 0.50     | 0.71      | 0.75      | 0.78      |
| `divrem` | 0.40     | 0.48     | 0.53     | 0.73      | 0.88      | 1.13      |

NBig matches or beats `BigInt` across the whole 128–4096 bit range for
`+`/`-`/`*`, and is within ~1.15× for `divrem` at the top of the range.

Kernel-level benchmarks against GMP's `__gmpn_*` functions directly live
alongside this one in `bench/` (see `bench/bench_kernels.jl`,
`bench/bench_mul.jl`, `bench/bench_divrem.jl`, etc.).
