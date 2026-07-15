# NativeBigInt.jl

Pure-Julia arbitrary-precision integer type (`NBig`) targeting GMP-competitive
performance from ~100 bits up: faster than GMP for `+`/`-`/`*`/`divrem`
through ~4k bits, with multiplication pulling well ahead again above ~14k
bits thanks to a floating-point NTT; within ~25% of GMP for `gcd`/`gcdx`
across the range (subquadratic HGCD above ~19k bits, ahead from ~16k end to
end); and ahead for `powermod` from ~16k bits (NTT-backed Barrett
reduction); and within ~15% of GMP for `isqrt` across the range (ahead below
~512 bits and above ~32k) — see the benchmark table below.
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
  the higher degree Toom algorithms as well). `mullo!`/`sqrlo!` are exact low
  short products (result mod β^k): truncated paired-row basecases at
  0.55–0.75× the full product deep into the Karatsuba range, a Mulders split
  (~0.69k full low block + two recursive cross products) above
  `MULLO_BASECASE_THRESHOLD`/`SQRLO_BASECASE_THRESHOLD`, and plain `mul!` +
  discard once the balanced product reaches the NTT
  (`bench/bench_kernels.jl mullo`).
- **Division (`src/div.jl`):** multi-limb `divrem!` — Knuth Algorithm D over
  `divrem_bc!` below ~100 limbs, GMP-`dcpi1`-style divide-and-conquer division
  (recursive 2n/n blocks over `mul!`, so it inherits Karatsuba and the NTT;
  0.82–0.96× GMP's `mpn_tdiv_qr` from the crossover through at least 2048
  limbs) above. `divappr!` is the quotient-only variant: a one-sided
  approximate quotient (never below the true one, within `DIVAPPR_ERR = 32`
  ulps) that skips all remainder work — a triangle-truncated schoolbook
  basecase (`divappr_bc!`, per-row divisor truncation) and a dc recursion
  that peels the top quotient half exactly and recurses approximately on the
  bottom with a truncated divisor.
- **Algorithms (`src/algorithms.jl`):** Karatsuba sqrt (Zimmermann), with a
  root-only top level for `isqrt`: above 16 top-level quotient limbs (~4k
  bits) the division runs as `divappr!` with one guard limb and a mantissa
  interval certificate settles whether the candidate root is exact or one
  too big — no remainder, no final square — reconstructing the remainder
  with one mul only in the ambiguous band (~2⁻⁵⁷ of inputs, plus perfect
  squares); below that the exact division's remainder feeds cheap positivity
  bounds that skip the final remainder square/subtract. Every level divides
  its halved numerator by the top of the root buffer directly
  (⌊N/2S′⌋ = ⌊(N>>1)/S′⌋ with U = 2U₁+ε): S′ is already normalized, so
  there is no divisor construction and no per-level renormalization shift,
  and the division engines run on the disposable numerator in place (no
  defensive copy; the remainder lands where the numerator was); `powermod` by
  sliding-window exponentiation, reducing each product with Montgomery
  `redc!` (`src/montgomery.jl`, odd moduli) or `divrem!` (even) at small
  sizes, and above per-parity thresholds (~68 limbs odd, ~240 even —
  `src/barrett.jl`, benchmark-tuned) with a plain-domain Barrett reduction
  (HAC 14.42) whose two per-product multiplies ride `mul!` (so
  Karatsuba/NTT) and whose reciprocal is one `divrem!` per modulus;
  radix conversion for `string`/`parse` — power-of-two bases pack/
  unpack bit windows directly (O(n)); other bases use per-limb `divrem_1!` /
  Horner below `STR_DC_THRESHOLD` (40 limbs, only coarsely tuned) and a
  recursive split/combine around a `bb^(2^i)` power tree above, routing through
  `divrem!`/`mul!` for O(M(n) log n) (the recursion lives in `src/nbig.jl`).
- **gcd (`src/gcd.jl`):** Lehmer gcd/gcdext (Knuth Algorithm L) on 126-bit
  leading windows — two bracket-verified single-word phases per window
  (hgcd2-flavoured), one fused matrix pass over the operands, a full
  division step when a window stalls, and for gcdx a V-cofactor pair
  carried in lockstep. Above ~300 limbs (gcd; ~250 for gcdx,
  `bench/bench_kernels.jl gcd`) a subquadratic HGCD layer in the style of GMP's
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
  approximation. Radix-4 DIF forward; the inverse direction is the
  transposed forward network (reverse stage order, the same forward twiddle
  tables — no inverse twiddles or 1/N pre-scale pass exist), which hands
  the unpack N·c index-reversed with the 1/N folded into unpack's
  descending read. Transform lengths m·2^k for
  m ∈ {1, 3, 5} (Winograd radix-3 and radix-5 passes; the heavier odd
  multipliers — m = 15 and the symmetric-pair radix-17 family
  m ∈ {17, 51, 85, 255} — join from m·2^14 points, where their
  cache-blocked odd passes and tighter padding beat one monolithic pow-2
  transform) keeping zero-padding waste
  ≤ ~25% (≤ ~18% once the 17 family is admitted), in-register shuffle
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
Shapes: `divrem` is 2n/n; `gcd`/`gcdx` operands share a planted n/2-bit
factor; `powermod` uses an n-bit modulus and a 512-bit-capped exponent.

| op | 128 | 256 | 512 | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|---|---|---|
| `+` | 0.51 | 0.55 | 0.92 | 0.60 | 0.88 | 0.75 | 1.02 | 1.48 | 1.21 |
| `-` | 0.53 | 0.50 | 0.89 | 0.60 | 0.83 | 0.88 | 1.17 | 1.82 | 1.55 |
| `*` | 0.50 | 0.61 | 0.87 | 0.87 | 1.12 | 1.19 | 1.12 | 0.84 | 0.65 |
| `divrem` | 0.41 | 0.37 | 0.66 | 0.86 | 1.05 | 0.82 | 0.88 | 1.04 | 1.10 |
| `gcd` | 1.04 | 1.11 | 1.09 | 1.08 | 1.16 | 1.09 | 1.06 | 0.99 | 0.81 |
| `gcdx` | 1.01 | 0.99 | 1.25 | 1.25 | 1.25 | 1.22 | 1.12 | 0.97 | 0.79 |
| `isqrt` | 0.81 | 0.96 | 1.14 | 1.09 | 1.12 | 1.14 | 1.07 | 1.05 | 0.94 |
| `powermod` (odd) | 2.00 | 1.14 | 1.95 | 1.67 | 1.36 | 1.33 | 1.48 | 1.07 | 0.76 |
| `powermod` (even) | 2.05 | 2.31 | 1.98 | 1.82 | 1.15 | 1.11 | 1.36 | 1.06 | 0.77 |

The broad shape: the core ring ops (`+`/`-`/`*`/`divrem`) beat GMP through
~4k bits and `*` pulls ahead again from ~16k as the fp NTT takes over
(`+`/`-` drift behind at 8k+, where both sides are memory-bound and GMP's
in-place reallocation wins). `gcd`/`gcdx` sit within ~25% throughout and
lead from ~16k. `powermod` pays GMP's assembly-Montgomery tax at small
sizes, reaches parity around 16k bits, and leads by ~25% at 32k where
Barrett reduction rides the NTT. `isqrt` sits within ~15% of GMP across the
range (per-input variance ±15%): the root-only divappr top level plus the
normalized-divisor level step (divide by S′, engines run in place) and
pool-sized allocations closed the former ~40% mid-range gap. An
approximate-Newton chain over short products (every level sqrlo+divappr) was
built and measured against this baseline and lost everywhere — the root-only
top already banks the big term, and reconstructing the child remainder costs
a half-size square the exact recursion gets as a byproduct — so it was
deleted (git history has it). Decimal `string`/`parse` are supported and
benchmarked in `bench_highlevel.jl` but slower than GMP's.

Above that, Karatsuba carries ~2k–14k bits at rough GMP parity (0.95–1.09×
against `__gmpn_mul`), and the two-prime fp NTT takes over at ~14k bits
already ahead (`bench/bench_kernels.jl mul`, AVX-512 machine; ratio is `mul!` /
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

Kernel- and threshold-level benchmarks against GMP's `__gmpn_*` functions
directly live alongside this one in `bench/bench_kernels.jl`, a family-selecting
driver (`micro`, `mul`, `sqr`, `div`, `kar`, `dc`, `gcd`, `mullo`, `barrett`,
`sqrt`); run it with no args for the family list.
