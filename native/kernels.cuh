#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>

__device__ inline float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }
__device__ inline float silu(float x) { return x * sigmoid(x); }
__device__ inline float silu_grad(float x) { float s = sigmoid(x); return s * (1.f + x * (1.f - s)); }
__device__ inline float warp_sum(float x) {
    for (int d = 16; d; d /= 2) x += __shfl_xor_sync(0xffffffff, x, d);
    return x;
}
__device__ inline float block_sum(float x) {
    __shared__ float sums[32];
    x = warp_sum(x);
    if ((threadIdx.x & 31) == 0) sums[threadIdx.x / 32] = x;
    __syncthreads();
    x = threadIdx.x < blockDim.x / 32 ? sums[threadIdx.x] : 0;
    if (threadIdx.x < 32) x = warp_sum(x);
    if (threadIdx.x == 0) sums[0] = x;
    __syncthreads();
    x = sums[0];
    __syncthreads();
    return x;
}
__device__ inline float block_max(float x) {
    __shared__ float sums[32];
    for (int d = 16; d; d /= 2) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, d));
    if ((threadIdx.x & 31) == 0) sums[threadIdx.x / 32] = x;
    __syncthreads();
    x = threadIdx.x < blockDim.x / 32 ? sums[threadIdx.x] : -INFINITY;
    if (threadIdx.x < 32)
        for (int d = 16; d; d /= 2) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, d));
    if (threadIdx.x == 0) sums[0] = x;
    __syncthreads();
    x = sums[0];
    __syncthreads();
    return x;
}

__global__ void bf16_to_float(const __nv_bfloat16 *x, float *y, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        y[i] = __bfloat162float(x[i]);
}
__global__ void float_to_bf16(const float *x, __nv_bfloat16 *y, size_t n) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)
        y[i]=__float2bfloat16_rn(x[i]);
}
__global__ void embedding_bf16(const __nv_bfloat16 *w,const uint32_t *ids,float *y,int n,int width) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n*width)y[i]=float(w[size_t(ids[i/width])*width+i%width]);
}
__global__ void tf32_split(const float *x,float *rounded,float *residual,size_t n) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
        // cuBLAS rounds TF32 operands to nearest even. Keep the discarded
        // component for a second tensor product with the exact BF16 weight.
        unsigned high;
        asm("cvt.rn.tf32.f32 %0, %1;" : "=r"(high) : "f"(x[i]));
        rounded[i]=__uint_as_float(high);
        residual[i]=x[i]-rounded[i];
    }
}
__global__ void embedding(const float *w, const uint32_t *ids, float *y, int n, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * width) y[i] = w[size_t(ids[i / width]) * width + i % width];
}
__global__ void add_forward(const float *a, const float *b, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}
__global__ void add_gradient(const float *dy, float *dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] += dy[i];
}
__global__ void slice_forward(const float *x, float *y, int n, int source_width,
                              int groups, int width, int source_group_width, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int row = i / (groups * width), h = i / width % groups, d = i % width;
        y[i] = x[row * source_width + h * source_group_width + offset + d];
    }
}
__global__ void slice_backward(const float *dy, float *dx, int n, int source_width,
                               int groups, int width, int source_group_width, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int row = i / (groups * width), h = i / width % groups, d = i % width;
        dx[row * source_width + h * source_group_width + offset + d] += dy[i];
    }
}
__global__ void norm_forward(const float *x, const float *w, float *y, int width, float offset, bool l2) {
    int row = blockIdx.x;
    float ss = 0;
    for (int d = threadIdx.x; d < width; d += blockDim.x) ss += x[row * width + d] * x[row * width + d];
    float inv = rsqrtf(block_sum(ss) / (l2 ? 1.f : float(width)) + 1e-6f);
    for (int d = threadIdx.x; d < width; d += blockDim.x)
        y[row * width + d] = x[row * width + d] * inv * (w ? w[d] + offset : 1.f);
}
__global__ void norm_backward(const float *x, const float *dy, const float *w, float *dx,
                              int width, float offset, bool l2) {
    int row = blockIdx.x;
    float ss = 0, dot = 0;
    for (int d = threadIdx.x; d < width; d += blockDim.x) {
        float a = x[row * width + d], g = dy[row * width + d] * (w ? w[d] + offset : 1.f);
        ss += a * a; dot += a * g;
    }
    float divisor = l2 ? 1.f : float(width);
    float inv = rsqrtf(block_sum(ss) / divisor + 1e-6f);
    float factor = block_sum(dot) * inv * inv / divisor;
    for (int d = threadIdx.x; d < width; d += blockDim.x)
        dx[row * width + d] += inv * (dy[row * width + d] * (w ? w[d] + offset : 1.f) - x[row * width + d] * factor);
}
__global__ void gated_forward(const float *a, const float *b, float *y, int n, bool use_silu,__nv_bfloat16 *packed=nullptr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float value=b[i] * (use_silu ? silu(a[i]) : sigmoid(a[i]));
        y[i]=value;if(packed)packed[i]=__float2bfloat16_rn(value);
    }
}
__global__ void gated_backward(const float *a, const float *b, const float *dy, float *da, float *db,
                               int n, bool use_silu,__nv_bfloat16 *da16=nullptr,__nv_bfloat16 *db16=nullptr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = sigmoid(a[i]);
        da[i] += dy[i] * b[i] * (use_silu ? silu_grad(a[i]) : s * (1.f - s));
        db[i] += dy[i] * (use_silu ? a[i] * s : s);
        if(da16)da16[i]=__float2bfloat16_rn(da[i]);
        if(db16)db16[i]=__float2bfloat16_rn(db[i]);
    }
}
__global__ void sigmoid_forward(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = sigmoid(x[i]);
}
__global__ void sigmoid_backward(const float *y, const float *dy, float *dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] += dy[i] * y[i] * (1.f - y[i]);
}
__global__ void decay_forward(const float *a, const float *alog, const float *bias, float *d, int n, int heads) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float z = a[i] + bias[i % heads];
        float softplus = fmaxf(z, 0.f) + log1pf(expf(-fabsf(z)));
        d[i] = expf(-expf(alog[i % heads]) * softplus);
    }
}
__global__ void decay_backward(const float *a, const float *alog, const float *bias,
                               const float *d, const float *dd, float *da, int n, int heads) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) da[i] += dd[i] * d[i] * -expf(alog[i % heads]) * sigmoid(a[i] + bias[i % heads]);
}
__global__ void conv_forward(const float *x, const float *w, float *z, float *y, int t, int width, int context=0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < t * width) {
        int row = i / width, c = i % width;
        float v = 0;
        for (int j = 0; j < 4; ++j) if (row%(context?context:t) + j >= 3) v += x[(row + j - 3) * width + c] * w[c * 4 + j];
        z[i] = v; y[i] = silu(v);
    }
}
__global__ void conv_backward(const float *z, const float *dy, const float *w, float *dx, int t, int width, int context=0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < t * width) {
        int row = i / width, c = i % width;
        float v = 0;
        for (int j = 0; j < 4; ++j) {
            int u = row + 3 - j;
            if (row%(context?context:t)+3-j < (context?context:t)) v += dy[u * width + c] * silu_grad(z[u * width + c]) * w[c * 4 + j];
        }
        dx[i] += v;
    }
}
__global__ void rope_forward(const float *x, float *y, int n, int heads, bool backward, int context=0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int d = i % 256, row = i / (heads * 256);
        if(context)row%=context;
        if (d < 64) {
            int j = d % 32, other = d < 32 ? i + 32 : i - 32;
            float angle = row * powf(1e7f, -float(j) / 32.f);
            float c = cosf(angle), s = sinf(angle) * (backward ? -1.f : 1.f);
            y[i] = c * x[i] + (d < 32 ? -s : s) * x[other];
        } else y[i] = x[i];
    }
}
__global__ void pack_heads(const float *x, float *y, int t, int input_heads, int heads, int width, int batch=1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch * heads * t * width) {
        int d = i % width, row = i / width % t, h = i / (t * width)%heads, seq=i/(t*width*heads);
        y[i] = x[((seq*t+row) * input_heads + h / (heads / input_heads)) * width + d];
    }
}
__global__ void unpack_heads(const float *x, float *y, int t, int output_heads, int heads, int width, bool add, int batch=1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch * heads * t * width) {
        int d = i % width, row = i / width % t, h = i / (t * width)%heads, seq=i/(t*width*heads);
        int out = ((seq*t+row) * output_heads + h / (heads / output_heads)) * width + d;
        if (add) atomicAdd(y + out, x[i]); else y[out] = x[i];
    }
}
__global__ void causal_softmax(float *p, int t) {
    int row = blockIdx.x % t, base = blockIdx.x * t;
    float hi = -INFINITY;
    for (int j = threadIdx.x; j <= row; j += blockDim.x) hi = fmaxf(hi, p[base + j]);
    hi = block_max(hi);
    float sum = 0;
    for (int j = threadIdx.x; j < t; j += blockDim.x) {
        float v = j <= row ? expf(p[base + j] - hi) : 0.f;
        p[base + j] = v; sum += v;
    }
    sum = block_sum(sum);
    for (int j = threadIdx.x; j < t; j += blockDim.x) p[base + j] /= sum;
}
__global__ void softmax_backward(const float *p, float *dp, int t) {
    int base = blockIdx.x * t;
    float dot = 0;
    for (int j = threadIdx.x; j < t; j += blockDim.x) dot += p[base + j] * dp[base + j];
    dot = block_sum(dot);
    for (int j = threadIdx.x; j < t; j += blockDim.x) dp[base + j] = p[base + j] * (dp[base + j] - dot) / 16.f;
}

// Recurrent Gated DeltaNet. One block per head, one lane per value dimension.
// Each lane owns an entire column of the 128x128 recurrent state in registers.
__global__ void delta_forward(const float *q, const float *k, const float *v, const float *d,
                              const float *beta, float *y, float *states, int t, int heads) {
    int h = blockIdx.x, j = threadIdx.x;
    float state[128];
    #pragma unroll
    for (int i = 0; i < 128; ++i) state[i] = 0;
    for (int row = 0; row < t; ++row) {
        int base = (row * heads + h) * 128;
        float decay = d[row * heads + h], dot = 0;
        #pragma unroll 1
        for (int i = 0; i < 128; ++i) { state[i] *= decay; dot += k[base + i] * state[i]; }
        float error = (v[base + j] - dot) * beta[row * heads + h], out = 0;
        #pragma unroll 1
        for (int i = 0; i < 128; ++i) {
            state[i] += k[base + i] * error;
            out += q[base + i] * state[i];
            if (states) states[(size_t(row * heads + h) * 128 + i) * 128 + j] = state[i];
        }
        y[base + j] = out * 0.08838834764831845f;
    }
}
__global__ void delta_backward(const float *q, const float *k, const float *v, const float *d,
                               const float *beta, const float *states, const float *dy,
                               float *dq, float *dk, float *dv, float *dd, float *db,
                               int t, int heads) {
    int h = blockIdx.x, j = threadIdx.x;
    float adj[128];
    #pragma unroll
    for (int i = 0; i < 128; ++i) adj[i] = 0;
    for (int row = t - 1; row >= 0; --row) {
        int base = (row * heads + h) * 128;
        size_t at = size_t(row * heads + h) * 128 * 128;
        float decay = d[row * heads + h], b = beta[row * heads + h];
        float output_grad = dy[base + j] * 0.08838834764831845f;
        float memory = 0, delta_grad = 0;
        #pragma unroll 1
        for (int i = 0; i < 128; ++i) {
            float prev = row ? states[at - size_t(heads) * 128 * 128 + i * 128 + j] : 0;
            memory += k[base + i] * prev * decay;
            adj[i] += q[base + i] * output_grad;
            delta_grad += k[base + i] * adj[i];
        }
        float error = v[base + j] - memory, value_grad = b * delta_grad, decay_grad = 0;
        dv[base + j] += value_grad;
        float beta_grad = warp_sum(delta_grad * error);
        if ((j & 31) == 0) atomicAdd(db + row * heads + h, beta_grad);
        #pragma unroll 1
        for (int i = 0; i < 128; ++i) {
            float prev = row ? states[at - size_t(heads) * 128 * 128 + i * 128 + j] : 0;
            float q_grad = warp_sum(output_grad * states[at + i * 128 + j]);
            float k_grad = warp_sum(adj[i] * b * error - prev * decay * value_grad);
            if ((j & 31) == 0) { atomicAdd(dq + base + i, q_grad); atomicAdd(dk + base + i, k_grad); }
            float memory_grad = adj[i] - k[base + i] * value_grad;
            decay_grad += memory_grad * prev;
            adj[i] = decay * memory_grad;
        }
        decay_grad = warp_sum(decay_grad);
        if ((j & 31) == 0) atomicAdd(dd + row * heads + h, decay_grad);
    }
}
template<class Gradient> __device__ void cross_entropy_impl(const float *logits, Gradient *grad, const uint32_t *target,
                              float *loss, int vocab, float scale) {
    int row = blockIdx.x;
    const float *x = logits + size_t(row) * vocab;
    float high = -INFINITY;
    for (int j = threadIdx.x; j < vocab; j += blockDim.x) high = fmaxf(high, x[j]);
    high = block_max(high);
    float sum = 0;
    for (int j = threadIdx.x; j < vocab; j += blockDim.x) sum += expf(x[j] - high);
    sum = block_sum(sum);
    if (threadIdx.x == 0) loss[row] = high + logf(sum) - x[target[row]];
    if (grad)
        for (int j = threadIdx.x; j < vocab; j += blockDim.x)
            grad[size_t(row) * vocab + j] = Gradient(scale * (expf(x[j] - high) / sum - float(j == target[row])));
}
__global__ void cross_entropy(const float *logits,float *grad,const uint32_t *target,float *loss,int vocab,float scale) {
    cross_entropy_impl(logits,grad,target,loss,vocab,scale);
}
__global__ void cross_entropy_bf16(const float *logits,__nv_bfloat16 *grad,const uint32_t *target,float *loss,int vocab,float scale) {
    cross_entropy_impl(logits,grad,target,loss,vocab,scale);
}
__global__ void gradient_norm(const float *g, float *sum, size_t n) {
    float s = 0;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        s += g[i] * g[i];
    s = block_sum(s);
    if (threadIdx.x == 0) atomicAdd(sum, s);
}
__global__ void adamw(float *p, const float *g, float *m, float *v, size_t n,
                     float lr, float clip, float correct1, float correct2) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        float x = g[i] * clip;
        m[i] = 0.9f * m[i] + 0.1f * x;
        v[i] = 0.999f * v[i] + 0.001f * x * x;
        p[i] -= lr * (m[i] / correct1 / (sqrtf(v[i] / correct2) + 1e-8f) + 0.01f * p[i]);
    }
}
