# NativeBigInt.jl

Pure-Julia arbitrary-precision integer type (`NBig`) targeting GMP-competitive
performance from ~100 bits up: within ~1.3× of GMP through the Karatsuba
range, and faster than GMP above ~33k bits thanks to a floating-point NTT
multiplication.
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
- **Multiplication (`src/mul.jl`):** subtractive Karatsuba (threshold ~29
  limbs, benchmark-tuned) with a general unbalanced-operand path; `mul!`/
  `sqr!` hand off to the fp NTT at ~336 balanced limbs (~400 for squaring).
  Toom-3 used to sit in between, but the fp NTT squeezed its winning band
  down to 240–340 limbs and it was removed.
- **Algorithms (`src/algorithms.jl`):** multi-limb `divrem!` (Knuth
  Algorithm D basecase over `divrem_bc!`); Karatsuba sqrt; power by repeated
  squaring; radix conversion for `string`/`parse` (per-limb `divrem_1!` for
  small values, divide-and-conquer for large).
- **fp NTT multiplication (`src/fpntt.jl`):** the main large-size engine —
  a number-theoretic transform over GF(p), p = 2^49 − 2^33 + 1, computed
  entirely in `Float64` in the style of FLINT's `fft_small`. Products use
  the FMA error-free transform (`h = x*w; l = fma(x, w, -h)` captures the
  exact 98-bit product with no carry chains), reduction is a Barrett
  quotient via the magic-constant round (p < 2^49 keeps every quotient
  below the 2^51 exactness bound), and twiddles are stored as Shoup-style
  `(w, w/p)` pairs so the quotient multiply runs in parallel with the
  product. Butterfly adds run unreduced with one reduction per butterfly
  closing the lazy bounds, so the hot loop is pure mul/fma/add traffic —
  ~2.3× faster per point than the integer engine, and every operation is
  exactly correct by rounding-error analysis, not approximation. Radix-4
  DIF/DIT, transform lengths m·2^k for m ∈ {1, 3, 5, 15} (Winograd
  radix-3, radix-5) keeping zero-padding waste ≤ ~25%, in-register shuffle
  butterflies for sub-vector-width stages, and squaring with one forward
  transform instead of two.
- **Integer NTT multiplication (`src/ntt.jl`):** the previous engine, over
  the Goldilocks field GF(2^64 − 2^32 + 1) (multiplication-free reduction,
  shift-only fourth-root rotation i = 2^48, same transform structure).
  Above ~131k limbs per operand it takes over again: the fp engine's
  single-prime coefficient bound forces its chunk width down as operands
  grow, while the 64-bit prime's bound decays more slowly. (A two-prime
  CRT extension of the fp engine would reclaim that range.) It also serves
  as the known-good integer-domain cross-check for the fp engine.
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

Above that, Karatsuba carries ~2k–21k bits and the fp NTT takes over from
~21k (`bench/bench_mul.jl`, AVX-512 machine; ratio is `mul!` / `__gmpn_mul`
on two equal operands of the given bit size):

| bits    | 33k  | 49k  | 66k  | 98k  | 131k | 262k | 524k | 2.1M | 16.8M | 268M |
|---------|------|------|------|------|------|------|------|------|-------|------|
| `*`     | 1.00 | 0.86 | 0.77 | 0.64 | 0.56 | 0.56 | 0.48 | 0.45 | 0.57  | 0.72 |

GMP parity now lands at ~33k bits — squarely inside GMP's hand-tuned
Toom-4/6.5 assembly range — and the fp NTT leads by ~1.8–2.2× from ~100k
bits through 2.1M. Above ~8M bits the single-prime fp chunk width has
decayed enough that dispatch returns to the integer Goldilocks NTT (the
16.8M and 268M columns), which holds a ~1.4× lead at the largest sizes
measured. Squaring crosses over at the same points with a slightly larger
lead. Asymptotically the fixed-prime NTT's chunk width shrinks as operands
grow, so Schönhage–Strassen would win again somewhere around 10^11 bits —
beyond both memory and the field's 2-adicity limit, so it never matters in
practice.

Kernel-level benchmarks against GMP's `__gmpn_*` functions directly live
alongside this one in `bench/` (see `bench/bench_kernels.jl`,
`bench/bench_mul.jl`, `bench/bench_divrem.jl`, etc.).
