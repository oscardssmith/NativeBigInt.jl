# NativeBigInt.jl

Pure-Julia arbitrary-precision integer type (`NBig`) targeting GMP-competitive
performance from ~100 bits up: within ~1.2× of GMP through the Karatsuba
range, and faster than GMP above ~100k bits thanks to an NTT multiplication.
Requires a recent Julia (uses `Memory{UInt64}`).

## Algorithms

Three layers, mirroring GMP's mpn/mpz split:

- **Kernels (`src/kernels/`):** sign-free limb-vector primitives —
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
- **NTT multiplication (`src/ntt.jl`):** above ~1800 combined limbs,
  `mul!`/`sqr!` dispatch to a number-theoretic transform over the Goldilocks
  field GF(2^64 − 2^32 + 1). Radix-4 DIF/DIT with fully vectorized
  butterflies (the 64×64 products are assembled from widening 32×32
  multiplies, so the hot path vectorizes portably on AVX2/AVX-512/NEON, and
  the reduction is multiplication-free since 2^64 ≡ 2^32 − 1); the fourth-root
  rotation i = 2^48 is shift-only; sub-vector-width stages run through
  in-register shuffle butterflies. Transform lengths m·2^k for m ∈
  {1, 3, 5, 15} (Winograd radix-3, radix-5) keep zero-padding waste ≤ ~25%,
  and pack/unpack stream branch-free through a 128-bit accumulator.
  Squaring uses one forward transform instead of two.
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

Above that, the NTT takes over (`bench/bench_mul.jl`, AVX-512 machine;
ratio is NBig `*` / BigInt `*`):

| bits    | 33k  | 66k  | 131k | 262k | 524k | 2.1M | 16.8M | 268M |
|---------|------|------|------|------|------|------|-------|------|
| `*`     | 1.12 | 1.15 | 0.93 | 0.85 | 0.81 | 0.69 | 0.64  | 0.74 |

GMP parity lands around ~100k bits, and the lead holds at roughly 1.3–1.5×
through the largest sizes measured (268M bits). Squaring crosses over at the
same point with a slightly larger lead. Asymptotically the fixed-prime NTT's
chunk width shrinks as operands grow, so Schönhage–Strassen would win again
somewhere around 10^11 bits — beyond both memory and the field's 2-adicity
limit, so it never matters in practice.

Kernel-level benchmarks against GMP's `__gmpn_*` functions directly live
alongside this one in `bench/` (see `bench/bench_kernels.jl`,
`bench/bench_mul.jl`, `bench/bench_divrem.jl`, etc.).
