// Transcription of llama.cpp's AVX2 Q8_0xQ8_0 / Q4_0xQ8_0 vec_dot loops (single accumulator, one block per iteration,
// sign trick + maddubs + madd) and the equivalent 4-row variants, for a C-vs-Rust codegen comparison. NOT the llama.cpp binary.
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef struct { uint16_t d; int8_t qs[32]; } bq8;
typedef struct { uint16_t d; uint8_t qs[16]; } bq4;
static inline float h2f(uint16_t h) { return _cvtsh_ss(h); }
static inline float hsum8(__m256 x) {
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}
static inline __m256 mul_sum_i8_pairs_float(__m256i x, __m256i y) {
    __m256i ax = _mm256_sign_epi8(x, x), sy = _mm256_sign_epi8(y, x);
    __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_set1_epi16(1), dot));
}
static inline __m256i bytes_from_nibbles_32(const uint8_t *p) {
    __m128i tmp = _mm_loadu_si128((const __m128i *)p);
    __m256i bytes = _mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp);
    return _mm256_and_si256(_mm256_set1_epi8(0xF), bytes);
}
__attribute__((noinline)) float dot_q8(int nb, const bq8 *x, const bq8 *y) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        __m256 d = _mm256_set1_ps(h2f(x[i].d) * h2f(y[i].d));
        __m256 q = mul_sum_i8_pairs_float(_mm256_loadu_si256((const __m256i *)x[i].qs), _mm256_loadu_si256((const __m256i *)y[i].qs));
        acc = _mm256_fmadd_ps(d, q, acc);
    }
    return hsum8(acc);
}
__attribute__((noinline)) float dot_q4(int nb, const bq4 *x, const bq8 *y) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        __m256 d = _mm256_set1_ps(h2f(x[i].d) * h2f(y[i].d));
        __m256i qx = _mm256_sub_epi8(bytes_from_nibbles_32(x[i].qs), _mm256_set1_epi8(8));
        __m256 q = mul_sum_i8_pairs_float(qx, _mm256_loadu_si256((const __m256i *)y[i].qs));
        acc = _mm256_fmadd_ps(d, q, acc);
    }
    return hsum8(acc);
}
__attribute__((noinline)) void gemv_q8(int rows, int nb, const bq8 *w, const bq8 *y, float *out) { for (int r = 0; r < rows; ++r) out[r] = dot_q8(nb, w + (size_t)r * nb, y); }
__attribute__((noinline)) void gemv_q4(int rows, int nb, const bq4 *w, const bq8 *y, float *out) { for (int r = 0; r < rows; ++r) out[r] = dot_q4(nb, w + (size_t)r * nb, y); }
__attribute__((noinline)) void gemv4_q8(int rows, int nb, const bq8 *w, const bq8 *y, float *out) {
    for (int r = 0; r < rows; r += 4) {
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        const bq8 *w0 = w + (size_t)r * nb, *w1 = w0 + nb, *w2 = w1 + nb, *w3 = w2 + nb;
        for (int i = 0; i < nb; ++i) {
            __m256i qy = _mm256_loadu_si256((const __m256i *)y[i].qs); float dy = h2f(y[i].d);
            a0 = _mm256_fmadd_ps(_mm256_set1_ps(h2f(w0[i].d) * dy), mul_sum_i8_pairs_float(_mm256_loadu_si256((const __m256i *)w0[i].qs), qy), a0);
            a1 = _mm256_fmadd_ps(_mm256_set1_ps(h2f(w1[i].d) * dy), mul_sum_i8_pairs_float(_mm256_loadu_si256((const __m256i *)w1[i].qs), qy), a1);
            a2 = _mm256_fmadd_ps(_mm256_set1_ps(h2f(w2[i].d) * dy), mul_sum_i8_pairs_float(_mm256_loadu_si256((const __m256i *)w2[i].qs), qy), a2);
            a3 = _mm256_fmadd_ps(_mm256_set1_ps(h2f(w3[i].d) * dy), mul_sum_i8_pairs_float(_mm256_loadu_si256((const __m256i *)w3[i].qs), qy), a3);
        }
        out[r] = hsum8(a0); out[r + 1] = hsum8(a1); out[r + 2] = hsum8(a2); out[r + 3] = hsum8(a3);
    }
}
__attribute__((noinline)) void gemv4_q4(int rows, int nb, const bq4 *w, const bq8 *y, float *out) {
    __m256i off = _mm256_set1_epi8(8);
    for (int r = 0; r < rows; r += 4) {
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        const bq4 *w0 = w + (size_t)r * nb, *w1 = w0 + nb, *w2 = w1 + nb, *w3 = w2 + nb;
        for (int i = 0; i < nb; ++i) {
            __m256i qy = _mm256_loadu_si256((const __m256i *)y[i].qs); float dy = h2f(y[i].d);
            a0 = _mm256_fmadd_ps(_mm256_set1_ps(h2f(w0[i].d) * dy), mul_sum_i8_pairs_float(_mm256_sub_epi8(bytes_from_nibbles_32(w0[i].qs), off), qy), a0);
            a1 = _mm256_fmadd_ps(_mm256_set1_ps(h2f(w1[i].d) * dy), mul_sum_i8_pairs_float(_mm256_sub_epi8(bytes_from_nibbles_32(w1[i].qs), off), qy), a1);
            a2 = _mm256_fmadd_ps(_mm256_set1_ps(h2f(w2[i].d) * dy), mul_sum_i8_pairs_float(_mm256_sub_epi8(bytes_from_nibbles_32(w2[i].qs), off), qy), a2);
            a3 = _mm256_fmadd_ps(_mm256_set1_ps(h2f(w3[i].d) * dy), mul_sum_i8_pairs_float(_mm256_sub_epi8(bytes_from_nibbles_32(w3[i].qs), off), qy), a3);
        }
        out[r] = hsum8(a0); out[r + 1] = hsum8(a1); out[r + 2] = hsum8(a2); out[r + 3] = hsum8(a3);
    }
}
