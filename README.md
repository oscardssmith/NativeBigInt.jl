# NativeBigInt.jl

Pure-Julia arbitrary-precision integer type (`NBig`) targeting GMP-competitive
performance from ~100 bits up: faster than GMP for `+`/`-`/`*` overall
(multiplication pulls well ahead above ~14k bits thanks to a floating-point
NTT), at or ahead of GMP for `divrem` across the range (schoolbook below
~8k bits, divide-and-conquer above riding the NTT multiplies), and
competitive with GMP for `gcd`/`gcdx` across the range (subquadratic HGCD
above ~19k bits, ahead of GMP's `mpn_gcd` from ~40k bits up), and
competitive below ~4k bits for roots and friends, which are still quadratic
and fall behind above that.
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
  `sqr!` hand off to the fp NTT at ~224 balanced limbs. This NTT implementation
  benchmarks better than Toom-3 for all sizes (and thus presumably better than
  the higher degree Toom algorithms as well).
- **Division (`src/div.jl`):** multi-limb `divrem!` — Knuth Algorithm D over
  `divrem_bc!` below ~100 limbs, GMP-`dcpi1`-style divide-and-conquer division
  (recursive 2n/n blocks over `mul!`, so it inherits Karatsuba and the NTT;
  0.82–0.96× GMP's `mpn_tdiv_qr` from the crossover through at least 2048
  limbs) above.
- **Algorithms (`src/algorithms.jl`):** Karatsuba sqrt; power by repeated
  squaring; radix conversion for `string`/`parse` — power-of-two bases pack/
  unpack bit windows directly (O(n)); other bases use per-limb `divrem_1!` /
  Horner below `STR_DC_THRESHOLD` (40 limbs, only coarsely tuned) and a
  recursive split/combine around a `bb^(2^i)` power tree above, routing through
  `divrem!`/`mul!` for O(M(n) log n) (the recursion lives in `src/nbig.jl`).
- **gcd (`src/gcd.jl`):** Lehmer gcd/gcdext (Knuth Algorithm L) on 126-bit
  leading windows — two bracket-verified single-word phases per window
  (hgcd2-flavoured), one fused matrix pass over the operands, a full
  division step when a window stalls, and for gcdx a V-cofactor pair
  carried in lockstep. Above ~300 limbs (gcd; ~250 for gcdx,
  `bench/bench_gcd_thr.jl`) a subquadratic HGCD layer in the style of GMP's
  `mpn_hgcd` (Möller 2008) takes over: recursive half-gcd builds a 2×2
  matrix of det-+1 mpn cofactors whose products route through `mul!`, so
  gcd inherits Karatsuba and the NTT for an O(M(n) log n) total — ahead of
  both the Lehmer loop and GMP's `mpn_gcd` from ~600 limbs (1.2× faster at
  2000 limbs, and widening).
- **fp NTT multiplication (`src/fpntt.jl`):** the sole large-size engine —
  number-theoretic transforms computed entirely in `Float64` in the style
  of FLINT's `fft_small`, with the convolution recombined by CRT over two
  ~2^49 primes (p₁ = 2^49 − 2^33 + 1, p₂ = 255·2^41 + 1; working modulus
  p₁·p₂ ≈ 2^99 keeps the chunk width at ~41–44 bits across all practical
  sizes, Garner recombination in the unpack). Products use the FMA
  error-free transform (`h = x*w; l = fma(x, w, -h)` captures the exact
  98-bit product with no carry chains), reduction is a Barrett quotient
  via the magic-constant round (p < 2^49 keeps every quotient below the
  2^51 exactness bound), and twiddles are stored as Shoup-style `(w, w/p)`
  pairs so the quotient multiply runs in parallel with the product.
  Butterfly adds run unreduced with one reduction per butterfly closing
  the lazy bounds, so the hot loop is pure mul/fma/add traffic, and every
  operation is exactly correct by rounding-error analysis, not
  approximation. Radix-4 DIF/DIT, transform lengths m·2^k for
  m ∈ {1, 3, 5} (Winograd radix-3 and radix-5 passes; m = 15 joins above
  ~2^14 points, where its cache-blocked odd passes beat one monolithic
  pow-2 transform) keeping zero-padding waste ≤ ~25%, in-register shuffle
  butterflies for sub-vector-width stages, and squaring with one forward
  transform instead of two.
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
| `+`      | 0.51     | 0.47     | 0.49     | 0.50      | 0.71      | 0.81      |
| `-`      | 0.46     | 0.50     | 0.51     | 0.56      | 0.71      | 0.92      |
| `*`      | 0.42     | 0.69     | 0.70     | 0.85      | 0.78      | 0.86      |
| `divrem` | 0.39     | 0.45     | 0.69     | 0.71      | 0.98      | 0.88      |

NBig matches or beats `BigInt` across the whole 128–4096 bit range for
all four ops.

Above that, Karatsuba carries ~2k–14k bits at rough GMP parity (0.95–1.09×
against `__gmpn_mul`), and the two-prime fp NTT takes over at ~14k bits
already ahead (`bench/bench_mul.jl`, AVX-512 machine; ratio is `mul!` /
`__gmpn_mul` on two equal operands of the given bit size):

| bits    | 14k  | 16k  | 33k  | 49k  | 66k  | 98k  | 131k | 262k | 524k | 2.1M | 16.8M | 268M |
|---------|------|------|------|------|------|------|------|------|------|------|-------|------|
| `*`     | 0.90 | 0.78 | 0.61 | 0.58 | 0.43 | 0.42 | 0.35 | 0.41 | 0.37 | 0.32 | 0.31  | 0.39 |

Through the Karatsuba band NBig trades blows with GMP's hand-tuned Toom
assembly, alternating sides of parity within ±10%; the ~14k-bit dispatch
threshold is where the lead becomes durable, growing to ~2.3–3.2× from
~66k bits up, including the range where GMP has switched to its own
Schönhage–Strassen FFT. Squaring crosses over at the same point with a slightly
larger lead (0.89× at 14k bits, ~0.30× at 2.1M). The fixed-prime chunk
width shrinks slowly as operands grow (the ratio drifting up toward the
268M-bit column), but the two-prime working modulus keeps the engine ahead
through the largest sizes that fit in memory.

Kernel-level benchmarks against GMP's `__gmpn_*` functions directly live
alongside this one in `bench/` (see `bench/bench_kernels.jl`,
`bench/bench_mul.jl`, `bench/div_vs_gmp.jl`, `bench/bench_dc_thr.jl`, etc.).
