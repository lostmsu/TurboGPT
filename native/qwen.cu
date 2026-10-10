// Native Qwen3.5 text-backbone LoRA. CUDA/cuBLAS own the entire training path.
// TensorBoard serialization is reused from the original CUDA C++ TurboGPT.
#include "kernels.cuh"
#include "delta.cuh"
#include "tensorboard.h"
#include "experiment.h"
#include "revision.h"
#include <array>
#include <cublas_v2.h>
#include <cudaProfiler.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
constexpr size_t GiB = size_t(1024) * 1024 * 1024;
constexpr int Hidden = 1024, Vocab = 248320;
// Preserve the 512-token optimizer geometry, scaling sequences for long contexts.
constexpr int ReferenceContext = 512, ReferenceBatch = 128, MaxContext = 4096;
constexpr int UpdateTokens = ReferenceContext * ReferenceBatch, MicrobatchTokens = 8192;
int effective_batch(int context) { return UpdateTokens / std::max(context, ReferenceContext); }
int default_batch(int context, int total) {
    int batch = std::min(total, std::max(1, MicrobatchTokens / context));
    while (total % batch) --batch;
    return batch;
}
void cuda_check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void blas_check(cublasStatus_t e) { if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error " + std::to_string(int(e))); }
int blocks(size_t n) { return int((n + 255) / 256); }

struct Arena {
    char *memory = nullptr;
    size_t capacity, position = 0, peak = 0;
    explicit Arena(size_t bytes) : capacity(bytes) { cuda_check(cudaMalloc(&memory, bytes)); }
    ~Arena() { cudaFree(memory); }
    template<class T> T *alloc(size_t count) {
        position = (position + 255) & ~size_t(255);
        size_t bytes = count * sizeof(T);
        if (bytes > capacity - position) throw std::runtime_error("Native training arena exceeded its VRAM budget: needs " +
            std::to_string(double(position+bytes)/GiB) + " GiB, cap " + std::to_string(double(capacity)/GiB) + " GiB");
        T *out = reinterpret_cast<T *>(memory + position);
        position += bytes; peak = std::max(peak, position);
        return out;
    }
};
struct Blas {
    cublasHandle_t handle;
    cublasComputeType_t compute = CUBLAS_COMPUTE_32F_FAST_TF32;
    bool corrected_tf32=false,native_bf16=false;
    Arena *scratch_arena=nullptr;
    std::unordered_set<const float *> exact_tf32;
    Blas() { blas_check(cublasCreate(&handle));blas_check(cublasSetStream(handle,cudaStreamPerThread)); }
    ~Blas() { cublasDestroy(handle); }
    void workspace(Arena &arena,size_t bytes) {
        // One stream serializes all GEMMs, so graph nodes can share one explicit
        // workspace. This keeps cuBLAS graph scratch inside the VRAM budget.
        blas_check(cublasSetWorkspace(handle,arena.alloc<char>(bytes),bytes));
    }
    void gemm16(float *c,const __nv_bfloat16 *a,const __nv_bfloat16 *b,int m,int n,int k,
                bool ta=false,bool tb=false,float alpha=1,float beta=0,int batch=1) {
        // Actual BF16 operands select the native BF16 tensor-core kernels.
        blas_check(cublasGemmStridedBatchedEx(handle,tb?CUBLAS_OP_T:CUBLAS_OP_N,ta?CUBLAS_OP_T:CUBLAS_OP_N,
            n,m,k,&alpha,b,CUDA_R_16BF,tb?k:n,int64_t(k)*n,a,CUDA_R_16BF,ta?m:k,int64_t(m)*k,
            &beta,c,CUDA_R_32F,n,int64_t(m)*n,batch,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    }
    void frozen_gemm(float *c,const float *a,const float *weight,const __nv_bfloat16 *packed,int m,int n,int k,
                     bool ta=false,bool tb=false,float alpha=1,float beta=0) {
        if(!packed){gemm(c,a,weight,m,n,k,ta,tb,alpha,beta);return;}
        size_t mark=scratch_arena->position,count=size_t(m)*k;
        auto converted=scratch_arena->alloc<__nv_bfloat16>(count);
        float_to_bf16<<<std::min(blocks(count),8192),256>>>(a,converted,count);
        gemm16(c,converted,packed,m,n,k,ta,tb,alpha,beta);
        scratch_arena->position=mark;
    }
    void gemm(float *c, const float *a, const float *b, int m, int n, int k,
              bool ta = false, bool tb = false, float alpha = 1, float beta = 0, int batch = 1) {
        if(native_bf16) {
            size_t mark=scratch_arena->position,na=size_t(batch)*m*k,nb=size_t(batch)*n*k;
            auto ap=scratch_arena->alloc<__nv_bfloat16>(na),bp=scratch_arena->alloc<__nv_bfloat16>(nb);
            float_to_bf16<<<std::min(blocks(na),8192),256>>>(a,ap,na);
            float_to_bf16<<<std::min(blocks(nb),8192),256>>>(b,bp,nb);
            gemm16(c,ap,bp,m,n,k,ta,tb,alpha,beta,batch);
            scratch_arena->position=mark;return;
        }
        // Row-major C=A*B maps to column-major C'=B'*A'.
        auto multiply=[&](const float *ap,const float *bp,float accumulation,cublasComputeType_t arithmetic) {
        blas_check(cublasGemmStridedBatchedEx(handle,
            tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N,
            n, m, k, &alpha, bp, CUDA_R_32F, tb ? k : n, int64_t(k) * n,
            ap, CUDA_R_32F, ta ? m : k, int64_t(m) * k,
            &accumulation, c, CUDA_R_32F, n, int64_t(m) * n, batch,
            arithmetic, CUBLAS_GEMM_DEFAULT));
        };
        if(!corrected_tf32){multiply(a,b,beta,compute);return;}
        bool ae=exact_tf32.contains(a),be=exact_tf32.contains(b);
        if(!ae && !be){multiply(a,b,beta,CUBLAS_COMPUTE_32F);return;}
        if(ae && be){multiply(a,b,beta,CUBLAS_COMPUTE_32F_FAST_TF32);return;}
        size_t count=size_t(batch)*k*(be?m:n),mark=scratch_arena->position;
        float *rounded=scratch_arena->alloc<float>(count),*residual=scratch_arena->alloc<float>(count);
        // Explicit rounding also makes this correct when a small GEMM uses a
        // native FP32 fallback instead of tensor cores.
        tf32_split<<<std::min(blocks(count),8192),256>>>(be?a:b,rounded,residual,count);
        multiply(be?rounded:a,be?b:rounded,beta,CUBLAS_COMPUTE_32F_FAST_TF32);
        multiply(be?residual:a,be?b:residual,1,CUBLAS_COMPUTE_32F_FAST_TF32);
        scratch_arena->position=mark;
    }
};
struct Weight { float *data=nullptr; std::vector<int> shape; size_t count; bool bf16=false; __nv_bfloat16 *packed=nullptr; };
struct Linear {
    float *weight, *a, *b, *ga, *gb;
    int input, output, rank;
    float scale;
    __nv_bfloat16 *packed=nullptr;
};
struct Parameters {
    Arena &arena;
    std::map<std::string, Weight> weights;
    std::map<std::string, Linear> linears;
    float *values, *gradients, *moment1, *moment2;
    size_t count = 0, persistent;
    int rank;
    float alpha;
    Parameters(Arena &a, const std::string &file, const std::string &manifest, int r, float al, uint32_t seed,bool native_bf16=false)
        : arena(a), rank(r), alpha(al) {
        std::ifstream registry(manifest), input(file, std::ios::binary);
        if (!registry || !input) throw std::runtime_error("Cannot open model assets");
        std::string name, dtype;
        uint64_t offset; int ndim;
        while (registry >> name >> dtype >> offset >> ndim) {
            Weight w; w.count = 1;
            for (int i = 0, d; i < ndim; ++i) { registry >> d; w.shape.push_back(d); w.count *= d; }
            if(native_bf16 && ndim==2 && dtype=="BF16")w.packed=arena.alloc<__nv_bfloat16>(w.count);
            else w.data=arena.alloc<float>(w.count);
            input.seekg(offset);
            if (dtype == "BF16") {
                w.bf16=true;
                std::vector<uint16_t> host(w.count);
                input.read(reinterpret_cast<char *>(host.data()), host.size() * 2);
                if(w.packed)cuda_check(cudaMemcpy(w.packed,host.data(),host.size()*2,cudaMemcpyHostToDevice));
                else {
                    size_t mark = arena.position;
                    auto temporary = arena.alloc<__nv_bfloat16>(w.count);
                    cuda_check(cudaMemcpy(temporary, host.data(), host.size() * 2, cudaMemcpyHostToDevice));
                    bf16_to_float<<<std::min(blocks(w.count), 8192), 256>>>(temporary, w.data, w.count);
                    cuda_check(cudaDeviceSynchronize());
                    arena.position = mark;
                }
            } else if (dtype == "F32") {
                std::vector<float> host(w.count);
                input.read(reinterpret_cast<char *>(host.data()), host.size() * 4);
                cuda_check(cudaMemcpy(w.data, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
            } else throw std::runtime_error("Unsupported weight dtype " + dtype);
            if (!input) throw std::runtime_error("Truncated safetensors weight " + name);
            weights[name] = w;
            if (w.shape.size() == 2 && name.find("layers.") == 0 && name.find("_proj") != std::string::npos)
                count += size_t(rank) * (w.shape[0] + w.shape[1]);
        }
        if (weights.size() != 320 || weights.at("embed_tokens.weight").shape != std::vector<int>{Vocab, Hidden})
            throw std::runtime_error("Expected the complete Qwen3.5-0.8B text backbone (320 tensors)");
        values = arena.alloc<float>(count); gradients = arena.alloc<float>(count);
        moment1 = arena.alloc<float>(count); moment2 = arena.alloc<float>(count);
        cuda_check(cudaMemset(gradients, 0, count * 4));
        cuda_check(cudaMemset(moment1, 0, count * 4)); cuda_check(cudaMemset(moment2, 0, count * 4));
        std::vector<float> initial(count, 0); std::mt19937 rng(seed); size_t cursor = 0;
        for (auto &[key, w] : weights) {
            if (w.shape.size() != 2 || key.find("layers.") != 0 || key.find("_proj") == std::string::npos) continue;
            Linear l; l.weight = w.data;l.packed=w.packed; l.output = w.shape[0]; l.input = w.shape[1]; l.rank = rank;
            l.scale = alpha / rank; size_t na = size_t(rank) * l.input, nb = size_t(rank) * l.output;
            l.a = values + cursor; l.ga = gradients + cursor;
            std::uniform_real_distribution<float> random(-1.f / std::sqrt(float(l.input)), 1.f / std::sqrt(float(l.input)));
            for (size_t i = 0; i < na; ++i) initial[cursor + i] = random(rng);
            cursor += na; l.b = values + cursor; l.gb = gradients + cursor; cursor += nb;
            linears[key] = l;
        }
        if (linears.size() != 186) throw std::runtime_error("Expected 186 text projection adapters");
        cuda_check(cudaMemcpy(values, initial.data(), count * 4, cudaMemcpyHostToDevice));
        persistent = arena.position;
        std::cout << "{\"loaded_text_tensors\":" << weights.size() << ",\"adapter_projections\":" << linears.size()
                  << ",\"trainable_parameters\":" << count << ",\"persistent_gib\":" << double(persistent) / GiB << "}" << std::endl;
    }
    float *w(const std::string &name) { return weights.at(name).data; }
    void embed(const uint32_t *ids,float *out,int rows) {
        const auto &w=weights.at("embed_tokens.weight");
        if(w.packed)embedding_bf16<<<blocks(size_t(rows)*Hidden),256>>>(w.packed,ids,out,rows,Hidden);
        else embedding<<<blocks(size_t(rows)*Hidden),256>>>(w.data,ids,out,rows,Hidden);
    }
    void head(Blas &blas,float *out,const float *input,int rows,bool backward=false,float beta=0) {
        const auto &w=weights.at("embed_tokens.weight");
        blas.frozen_gemm(out,input,w.data,w.packed,rows,backward?Hidden:Vocab,backward?Vocab:Hidden,false,!backward,1,beta);
    }
};
#include "optimizers.cuh"
struct Tensor { float *data, *grad = nullptr; int rows, width; __nv_bfloat16 *packed=nullptr,*packed_grad=nullptr; size_t size() const { return size_t(rows) * width; } };
using T = Tensor *;
struct Graph {
    Arena &arena; Blas &blas; bool training;
    bool adapters = true;
    int context=0,batch=1;
    std::function<void(int,T)> observe_layer;
    std::vector<std::unique_ptr<Tensor>> tensors;
    std::vector<std::function<void()>> tape;
    Graph(Arena &a, Blas &b, bool train) : arena(a), blas(b), training(train) {}
    T tensor(int rows, int width) {
        auto x = std::make_unique<Tensor>(); x->rows = rows; x->width = width; x->data = arena.alloc<float>(x->size());
        T out = x.get(); tensors.push_back(std::move(x)); return out;
    }
    T view(float *data,int rows,int width,float *gradient=nullptr) {
        auto x=std::make_unique<Tensor>();x->data=data;x->rows=rows;x->width=width;x->grad=gradient;
        T out=x.get();tensors.push_back(std::move(x));return out;
    }
    float *grad(T x) {
        // Requesting a writable gradient invalidates an earlier packed view.
        // A fused producer can restore it after incorporating its contribution.
        x->packed_grad=nullptr;
        if (!x->grad) { x->grad = arena.alloc<float>(x->size()); cuda_check(cudaMemsetAsync(x->grad, 0, x->size() * 4,cudaStreamPerThread)); }
        return x->grad;
    }
    void backward() { for (auto i = tape.rbegin(); i != tape.rend(); ++i) (*i)(); cuda_check(cudaGetLastError()); }
    __nv_bfloat16 *pack(const float *data,size_t count) {
        auto out=arena.alloc<__nv_bfloat16>(count);
        float_to_bf16<<<std::min(blocks(count),8192),256>>>(data,out,count);return out;
    }
    __nv_bfloat16 *packed(T x) {
        // Forward tensor values are immutable. Share the conversion across
        // projections and retain it for their backward GEMMs.
        if(!x->packed)x->packed=pack(x->data,x->size());return x->packed;
    }
    T linear(T x, Linear l) {
        if (x->width != l.input) throw std::runtime_error("Projection shape mismatch");
        T y = tensor(x->rows, l.output), projected = tensor(x->rows, l.rank);
        if(blas.native_bf16) {
            auto xp=packed(x),wp=l.packed?l.packed:pack(l.weight,size_t(l.input)*l.output);
            blas.gemm16(y->data,xp,wp,x->rows,l.output,l.input,false,true);
            if(!adapters)return y;
            auto ap=pack(l.a,size_t(l.rank)*l.input),bp=pack(l.b,size_t(l.rank)*l.output);
            blas.gemm16(projected->data,xp,ap,x->rows,l.rank,l.input,false,true);
            auto sp=packed(projected);
            blas.gemm16(y->data,sp,bp,x->rows,l.output,l.rank,false,true,l.scale,1);
            if(training)tape.push_back([=,this] {
                if(!y->grad)return;
                float *gx=grad(x);size_t mark=arena.position;
                auto yp=y->packed_grad?y->packed_grad:pack(y->grad,y->size());auto gs=arena.alloc<float>(projected->size());
                blas.gemm16(gx,yp,wp,x->rows,l.input,l.output,false,false,1,1);
                blas.gemm16(l.gb,yp,sp,l.output,l.rank,x->rows,true,false,l.scale,1);
                blas.gemm16(gs,yp,bp,x->rows,l.rank,l.output,false,false,l.scale);
                auto gp=pack(gs,projected->size());
                blas.gemm16(l.ga,gp,xp,l.rank,l.input,x->rows,true,false,1,1);
                blas.gemm16(gx,gp,ap,x->rows,l.input,l.rank,false,false,1,1);
                arena.position=mark;
            });
            return y;
        }
        blas.frozen_gemm(y->data, x->data, l.weight,l.packed, x->rows, l.output, l.input, false, true);
        if (!adapters) return y;
        blas.gemm(projected->data, x->data, l.a, x->rows, l.rank, l.input, false, true);
        blas.gemm(y->data, projected->data, l.b, x->rows, l.output, l.rank, false, true, l.scale, 1);
        if (training) tape.push_back([=, this] {
            if (!y->grad) return;
            float *gx = grad(x);size_t mark=arena.position;
            float *gs = arena.alloc<float>(projected->size());
            blas.frozen_gemm(gx, y->grad, l.weight,l.packed, x->rows, l.input, l.output, false, false, 1, 1);
            blas.gemm(l.gb, y->grad, projected->data, l.output, l.rank, x->rows, true, false, l.scale, 1);
            blas.gemm(gs, y->grad, l.b, x->rows, l.rank, l.output, false, false, l.scale);
            blas.gemm(l.ga, gs, x->data, l.rank, l.input, x->rows, true, false, 1, 1);
            blas.gemm(gx, gs, l.a, x->rows, l.input, l.rank, false, false, 1, 1);
            arena.position=mark;
        });
        return y;
    }
    T add(T x, T z) {
        T y = tensor(x->rows, x->width);
        add_forward<<<blocks(y->size()), 256>>>(x->data, z->data, y->data, int(y->size()));
        if (training) tape.push_back([=, this] { if (y->grad) {
            add_gradient<<<blocks(y->size()), 256>>>(y->grad, grad(x), int(y->size()));
            add_gradient<<<blocks(y->size()), 256>>>(y->grad, grad(z), int(y->size()));
        }});
        return y;
    }
    T norm(T x, int group_width, float *weight = nullptr, float offset = 0, bool l2 = false) {
        T y = tensor(x->rows, x->width); int rows = int(x->size()) / group_width;
        int threads=group_width==128?128:256;
        norm_forward<<<rows, threads>>>(x->data, weight, y->data, group_width, offset, l2);
        if (training) tape.push_back([=, this] { if (y->grad)
            norm_backward<<<rows, threads>>>(x->data, y->grad, weight, grad(x), group_width, offset, l2);
        });
        return y;
    }
    T gated(T gate, T value, bool use_silu) {
        T y = tensor(gate->rows, gate->width);
        if(blas.native_bf16)y->packed=arena.alloc<__nv_bfloat16>(y->size());
        gated_forward<<<blocks(y->size()), 256>>>(gate->data, value->data, y->data, int(y->size()), use_silu,y->packed);
        if (training) tape.push_back([=, this] { if (y->grad) {
            auto dg=grad(gate),dv=grad(value);
            if(blas.native_bf16) {
                gate->packed_grad=arena.alloc<__nv_bfloat16>(gate->size());
                value->packed_grad=arena.alloc<__nv_bfloat16>(value->size());
            }
            gated_backward<<<blocks(y->size()), 256>>>(gate->data,value->data,y->grad,dg,dv,int(y->size()),use_silu,gate->packed_grad,value->packed_grad);
        }
        });
        return y;
    }
    T slice(T x, int groups, int width, int group_stride, int offset) {
        T y = tensor(x->rows, groups * width);
        slice_forward<<<blocks(y->size()), 256>>>(x->data, y->data, int(y->size()), x->width, groups, width, group_stride, offset);
        if (training) tape.push_back([=, this] { if (y->grad)
            slice_backward<<<blocks(y->size()), 256>>>(y->grad, grad(x), int(y->size()), x->width, groups, width, group_stride, offset);
        });
        return y;
    }
    T conv(T x, float *w) {
        T y = tensor(x->rows, x->width); float *pre = arena.alloc<float>(y->size());
        conv_forward<<<blocks(y->size()), 256>>>(x->data, w, pre, y->data, x->rows, x->width,context);
        if (training) tape.push_back([=, this] { if (y->grad)
            conv_backward<<<blocks(y->size()), 256>>>(pre, y->grad, w, grad(x), x->rows, x->width,context);
        });
        return y;
    }
    T sig(T x) {
        T y = tensor(x->rows, x->width);
        sigmoid_forward<<<blocks(y->size()), 256>>>(x->data, y->data, int(y->size()));
        if (training) tape.push_back([=, this] { if (y->grad)
            sigmoid_backward<<<blocks(y->size()), 256>>>(y->data, y->grad, grad(x), int(y->size()));
        }); return y;
    }
    T decay(T x, float *alog, float *bias) {
        T y = tensor(x->rows, x->width);
        decay_forward<<<blocks(y->size()), 256>>>(x->data, alog, bias, y->data, int(y->size()), x->width);
        if (training) tape.push_back([=, this] { if (y->grad)
            decay_backward<<<blocks(y->size()), 256>>>(x->data, alog, bias, y->data, y->grad, grad(x), int(y->size()), x->width);
        }); return y;
    }
    T rope(T x, int heads) {
        T y = tensor(x->rows, x->width);
        rope_forward<<<blocks(y->size()), 256>>>(x->data, y->data, int(y->size()), heads, false,context);
        if (training) tape.push_back([=, this] { if (y->grad) {
            float *gx=grad(x);size_t mark=arena.position;
            float *temporary = arena.alloc<float>(y->size());
            rope_forward<<<blocks(y->size()), 256>>>(y->grad, temporary, int(y->size()), heads, true,context);
            add_gradient<<<blocks(y->size()), 256>>>(temporary, gx, int(y->size()));
            arena.position=mark;
        }}); return y;
    }
    T delta(T q, T k, T v, T d, T beta) {
        T y = tensor(q->rows, q->width);
        int t=context?context:q->rows,chunks=(t+DeltaChunk-1)/DeltaChunk;
        float *states = training ? arena.alloc<float>(size_t(batch)*chunks*16*128*128) : nullptr;
        delta_tiled_forward<<<batch*16*8,128>>>(q->data,k->data,v->data,d->data,beta->data,y->data,states,t,16);
        if(training)tape.push_back([=,this]{if(y->grad){
            float *gq=grad(q),*gk=grad(k),*gv=grad(v),*gd=grad(d),*gb=grad(beta);
            size_t mark=arena.position;
            float *history=nullptr;
            float *errors=arena.alloc<float>(size_t(batch)*DeltaChunk*16*128);
            float2 *qk=arena.alloc<float2>(size_t(batch)*16*DeltaGroups*t*128);
            float2 *db=arena.alloc<float2>(size_t(batch)*16*DeltaGroups*t);
            delta_tiled_backward<<<batch*16*8,128>>>(q->data,k->data,v->data,d->data,beta->data,states,y->grad,
                gv,history,errors,qk,db,t,16);
            delta_reduce<<<blocks(q->size()),256>>>(qk,db,gq,gk,gd,gb,t,16,batch);
            arena.position=mark;
        }});return y;
    }
    T attention(T q, T k, T v) {
        int t = context?context:q->rows; size_t size = size_t(batch) * 8 * t * 256;
        float *qp = arena.alloc<float>(size), *kp = arena.alloc<float>(size), *vp = arena.alloc<float>(size);
        float *p = arena.alloc<float>(size_t(batch) * 8 * t * t), *out = arena.alloc<float>(size);
        pack_heads<<<blocks(size), 256>>>(q->data, qp, t, 8, 8, 256,batch);
        pack_heads<<<blocks(size), 256>>>(k->data, kp, t, 2, 8, 256,batch);
        pack_heads<<<blocks(size), 256>>>(v->data, vp, t, 2, 8, 256,batch);
        auto qp16=blas.native_bf16?pack(qp,size):nullptr,kp16=blas.native_bf16?pack(kp,size):nullptr;
        auto vp16=blas.native_bf16?pack(vp,size):nullptr;
        if(blas.native_bf16)blas.gemm16(p,qp16,kp16,t,t,256,false,true,1.f/16,0,batch*8);
        else blas.gemm(p, qp, kp, t, t, 256, false, true, 1.f / 16, 0, batch*8);
        causal_softmax<<<batch*8 * t, 256>>>(p, t);
        auto p16=blas.native_bf16?pack(p,size_t(batch)*8*t*t):nullptr;
        if(blas.native_bf16)blas.gemm16(out,p16,vp16,t,256,t,false,false,1,0,batch*8);
        else blas.gemm(out, p, vp, t, 256, t, false, false, 1, 0, batch*8);
        T y = tensor(q->rows, 2048);
        unpack_heads<<<blocks(size), 256>>>(out, y->data, t, 8, 8, 256, false,batch);
        if (training) tape.push_back([=, this] {
            if (!y->grad) return;
            float *dq=grad(q),*dk=grad(k),*dv=grad(v);size_t mark=arena.position;
            float *go = arena.alloc<float>(size), *gp = arena.alloc<float>(size_t(batch) * 8 * t * t);
            float *gq = arena.alloc<float>(size), *gk = arena.alloc<float>(size), *gv = arena.alloc<float>(size);
            pack_heads<<<blocks(size), 256>>>(y->grad, go, t, 8, 8, 256,batch);
            if(blas.native_bf16) {
                auto go16=pack(go,size);
                blas.gemm16(gp,go16,vp16,t,t,256,false,true,1,0,batch*8);
                blas.gemm16(gv,p16,go16,t,256,t,true,false,1,0,batch*8);
                softmax_backward<<<batch*8*t,256>>>(p,gp,t);
                auto gp16=pack(gp,size_t(batch)*8*t*t);
                blas.gemm16(gq,gp16,kp16,t,256,t,false,false,1,0,batch*8);
                blas.gemm16(gk,gp16,qp16,t,256,t,true,false,1,0,batch*8);
            } else {
            blas.gemm(gp, go, vp, t, t, 256, false, true, 1, 0, batch*8);
            blas.gemm(gv, p, go, t, 256, t, true, false, 1, 0, batch*8);
            softmax_backward<<<batch*8 * t, 256>>>(p, gp, t);
            blas.gemm(gq, gp, kp, t, 256, t, false, false, 1, 0, batch*8);
            blas.gemm(gk, gp, qp, t, 256, t, true, false, 1, 0, batch*8);
            }
            unpack_heads<<<blocks(size), 256>>>(gq, dq, t, 8, 8, 256, true,batch);
            unpack_heads<<<blocks(size), 256>>>(gk, dk, t, 2, 8, 256, true,batch);
            unpack_heads<<<blocks(size), 256>>>(gv, dv, t, 2, 8, 256, true,batch);
            arena.position=mark;
        }); return y;
    }
};

T layer_forward(Graph &g,Parameters &p,T x,int l) {
        std::string layer = "layers." + std::to_string(l) + ".";
        auto projection = [&](T input, const std::string &name) { return g.linear(input, p.linears.at(layer + name + ".weight")); };
        T normalized = g.norm(x, Hidden, p.w(layer + "input_layernorm.weight"), 1);
        T branch;
        if (l % 4 != 3) {
            T qkv = g.conv(projection(normalized, "linear_attn.in_proj_qkv"), p.w(layer + "linear_attn.conv1d.weight"));
            T q = g.norm(g.slice(qkv, 1, 2048, 6144, 0), 128, nullptr, 0, true);
            T k = g.norm(g.slice(qkv, 1, 2048, 6144, 2048), 128, nullptr, 0, true);
            T v = g.slice(qkv, 1, 2048, 6144, 4096);
            T beta = g.sig(projection(normalized, "linear_attn.in_proj_b"));
            T d = g.decay(projection(normalized, "linear_attn.in_proj_a"), p.w(layer + "linear_attn.A_log"), p.w(layer + "linear_attn.dt_bias"));
            T core = g.delta(q, k, v, d, beta);
            core = g.norm(core, 128, p.w(layer + "linear_attn.norm.weight"));
            T z = projection(normalized, "linear_attn.in_proj_z");
            branch = projection(g.gated(z, core, true), "linear_attn.out_proj");
        } else {
            T qg = projection(normalized, "self_attn.q_proj");
            T q = g.rope(g.norm(g.slice(qg, 8, 256, 512, 0), 256, p.w(layer + "self_attn.q_norm.weight"), 1), 8);
            T k = g.rope(g.norm(projection(normalized, "self_attn.k_proj"), 256, p.w(layer + "self_attn.k_norm.weight"), 1), 2);
            T v = projection(normalized, "self_attn.v_proj");
            T core = g.attention(q, k, v);
            branch = projection(g.gated(g.slice(qg, 8, 256, 512, 256), core, false), "self_attn.o_proj");
        }
        T middle = g.add(x, branch);
        T mlp_input = g.norm(middle, Hidden, p.w(layer + "post_attention_layernorm.weight"), 1);
        T gate = projection(mlp_input, "mlp.gate_proj"), up = projection(mlp_input, "mlp.up_proj");
        return g.add(middle, projection(g.gated(gate, up, true), "mlp.down_proj"));
}
T forward(Graph &g, Parameters &p, uint32_t *ids, int context) {
    if(!g.context)g.context=context;
    T x = g.tensor(context, Hidden);
    p.embed(ids,x->data,context);
    if(g.observe_layer)g.observe_layer(0,x);
    if(!g.training) {
        // Inference has no backward tape. Reuse one layer's scratch instead of
        // retaining all 24 layers, including six quadratic attention matrices.
        T next=g.tensor(context,Hidden);size_t workspace=g.arena.position;
        for(int l=0;l<24;++l) {
            g.arena.position=workspace;Graph layer(g.arena,g.blas,false);
            layer.context=g.context;layer.batch=g.batch;layer.adapters=g.adapters;
            T input=layer.view(x->data,context,Hidden),out=layer_forward(layer,p,input,l);
            if(g.observe_layer)g.observe_layer(l+1,out);
            cuda_check(cudaMemcpyAsync(next->data,out->data,x->size()*4,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
            std::swap(x,next);
        }
        g.arena.position=workspace;
        T normalized=g.norm(x,Hidden,p.w("norm.weight"),1);
        if(g.observe_layer)g.observe_layer(25,normalized);
        return normalized;
    }
    for (int l = 0; l < 24; ++l)x=layer_forward(g,p,x,l);
    return g.norm(x, Hidden, p.w("norm.weight"), 1);
}

struct Metrics {
    static constexpr std::array<int,8> Positions={2,3,5,17,65,257,1025,4097};
    std::array<double,8> position_nll{};
    std::array<uint64_t,8> position_count{};
    double nll = 0, last_nll = 0;
    uint64_t tokens = 0, bytes = 0, last_tokens = 0, last_bytes = 0;
    void add(const Metrics &x) { nll += x.nll; last_nll += x.last_nll; tokens += x.tokens;
        bytes += x.bytes; last_tokens += x.last_tokens; last_bytes += x.last_bytes;
        for(size_t i=0;i<Positions.size();++i){position_nll[i]+=x.position_nll[i];position_count[i]+=x.position_count[i];} }
    double loss() const { return nll / tokens; }
    double bpb() const { return nll / std::log(2.) / bytes; }
    double last_loss() const { return last_nll / last_tokens; }
    double last_bpb() const { return last_nll / std::log(2.) / last_bytes; }
};
Metrics collect_losses(const float *losses,const std::vector<uint32_t> &targets,
    const std::vector<uint32_t> &token_bytes,int context) {
    Metrics result;
    for(size_t i=0;i<targets.size();++i) {
        if(!std::isfinite(losses[i]))throw std::runtime_error("Non-finite cross-entropy");
        uint32_t bytes=token_bytes.at(targets[i]);if(!bytes)throw std::runtime_error("Target has no original-byte length");
        result.nll+=losses[i];++result.tokens;result.bytes+=bytes;
        if(i%context==context-1){result.last_nll+=losses[i];++result.last_tokens;result.last_bytes+=bytes;}
        for(size_t j=0;j<Metrics::Positions.size();++j)if(i%context==Metrics::Positions[j]) {
            result.position_nll[j]+=losses[i];++result.position_count[j];
        }
    }
    return result;
}
Metrics loss(Graph &g, Parameters &p, T hidden, uint32_t *targets,
             const std::vector<uint32_t> &host_targets, const std::vector<uint32_t> &token_bytes,
             int accumulation,float *pinned_losses=nullptr) {
    int context = hidden->rows, chunk = std::min(2048,context);
    if (g.training) g.grad(hidden);
    size_t temporary_mark = g.arena.position;
    float *logits = g.arena.alloc<float>(size_t(chunk) * Vocab);
    float *dlogits = g.training && !g.blas.native_bf16 ? g.arena.alloc<float>(size_t(chunk) * Vocab) : nullptr;
    auto packed_grad=g.training && g.blas.native_bf16 ? g.arena.alloc<__nv_bfloat16>(size_t(chunk)*Vocab) : nullptr;
    float *losses = g.arena.alloc<float>(context);
    for (int start = 0; start < context; start += chunk) {
        int rows = std::min(chunk, context - start);
        p.head(g.blas,logits,hidden->data+start*Hidden,rows);
        if(packed_grad) {
            // Write the gradient in its consuming GEMM's dtype directly.
            cross_entropy_bf16<<<rows,256>>>(logits,packed_grad,targets+start,losses+start,Vocab,1.f/(context*accumulation));
            g.blas.gemm16(hidden->grad+start*Hidden,packed_grad,p.weights.at("embed_tokens.weight").packed,
                          rows,Hidden,Vocab,false,false,1,1);
        } else {
            cross_entropy<<<rows, 256>>>(logits, dlogits, targets + start, losses+start, Vocab, 1.f / (context * accumulation));
        if (g.training)
            p.head(g.blas,hidden->grad+start*Hidden,dlogits,rows,true,1);
        }
    }
    Metrics result;
    if(pinned_losses)cuda_check(cudaMemcpyAsync(pinned_losses,losses,context*4,cudaMemcpyDeviceToHost,cudaStreamPerThread));
    else {
        std::vector<float> host(context);cuda_check(cudaMemcpy(host.data(),losses,context*4,cudaMemcpyDeviceToHost));
        result=collect_losses(host.data(),host_targets,token_bytes,g.context?g.context:context);
    }
    g.arena.position = temporary_mark;
    return result;
}

struct Tokens {
    std::ifstream file; uint64_t count;
    explicit Tokens(const std::string &path) : file(path, std::ios::binary) {
        if (!file) throw std::runtime_error("Cannot read tokens " + path);
        count = fs::file_size(path) / 4;
    }
    std::vector<uint32_t> block(uint64_t index, int context) {
        if (index + context >= count) throw std::runtime_error("Token block exceeds dataset");
        std::vector<uint32_t> data(context + 1); file.seekg(index * 4);
        file.read(reinterpret_cast<char *>(data.data()), data.size() * 4);
        if (!file) throw std::runtime_error("Truncated token file");
        for (uint32_t id : data) if (id >= Vocab) throw std::runtime_error("Invalid token ID");
        return data;
    }
};
struct Runner {
    Arena &arena; Blas &blas; Parameters &parameters;
    std::vector<uint32_t> byte_lengths;
    int context, accumulation;
    bool checkpoint_layers=true;
    std::vector<std::vector<uint32_t>> evaluation_inputs;
    bool cuda_graphs=true,warmed=false;
    cudaGraphExec_t graph_exec=nullptr;
    float *pinned_losses=nullptr;int graph_rows=0;
    Runner(Arena &a, Blas &b, Parameters &p, const std::string &lengths, int c, int accum)
        : arena(a), blas(b), parameters(p), context(c), accumulation(accum), byte_lengths(Vocab) {
        std::ifstream input(lengths, std::ios::binary);
        input.read(reinterpret_cast<char *>(byte_lengths.data()), Vocab * 4);
        if (!input) throw std::runtime_error("Missing token byte-length table");
    }
    ~Runner(){if(graph_exec)cudaGraphExecDestroy(graph_exec);if(pinned_losses)cudaFreeHost(pinned_losses);}
    Metrics batch(const std::vector<uint32_t> &tokens, bool training) {
        arena.position = parameters.persistent;
        int sequences=int(tokens.size())/(context+1),rows=sequences*context;
        if(sequences<1 || tokens.size()!=size_t(sequences)*(context+1))throw std::runtime_error("Invalid physical batch");
        std::vector<uint32_t> inputs(rows),targets(rows);
        for(int b=0;b<sequences;++b) {
            std::copy_n(tokens.begin()+b*(context+1),context,inputs.begin()+b*context);
            std::copy_n(tokens.begin()+b*(context+1)+1,context,targets.begin()+b*context);
        }
        uint32_t *ids = arena.alloc<uint32_t>(rows),*target=arena.alloc<uint32_t>(rows);
        cuda_check(cudaMemcpy(ids,inputs.data(),rows*4,cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(target,targets.data(),rows*4,cudaMemcpyHostToDevice));
        if(training && cuda_graphs) {
            if(!pinned_losses){cuda_check(cudaMallocHost(&pinned_losses,rows*4));graph_rows=rows;}
            if(graph_rows!=rows)throw std::runtime_error("Cached CUDA graph batch shape changed");
            if(graph_exec) {
                cuda_check(cudaGraphLaunch(graph_exec,cudaStreamPerThread));
                cuda_check(cudaStreamSynchronize(cudaStreamPerThread));
                return collect_losses(pinned_losses,targets,byte_lengths,context);
            }
        }
        bool capture=training && cuda_graphs && warmed;
        if(capture)cuda_check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
        Metrics metrics;
        if(training && checkpoint_layers) {
            size_t n=size_t(rows)*Hidden;
            float *saved=arena.alloc<float>(n*25),*adj[2]={arena.alloc<float>(n),arena.alloc<float>(n)};
            size_t workspace=arena.position;
            parameters.embed(ids,saved,rows);
            for(int layer=0;layer<24;++layer) {
                arena.position=workspace;Graph graph(arena,blas,false);graph.context=context;graph.batch=sequences;
                T x=graph.view(saved+layer*n,rows,Hidden);T out=layer_forward(graph,parameters,x,layer);
                cuda_check(cudaMemcpyAsync(saved+(layer+1)*n,out->data,n*4,cudaMemcpyDeviceToDevice));
            }
            arena.position=workspace;
            {
                Graph graph(arena,blas,true);graph.context=context;graph.batch=sequences;
                cuda_check(cudaMemsetAsync(adj[0],0,n*4,cudaStreamPerThread));
                T x=graph.view(saved+24*n,rows,Hidden,adj[0]);T hidden=graph.norm(x,Hidden,parameters.w("norm.weight"),1);
                metrics=loss(graph,parameters,hidden,target,targets,byte_lengths,accumulation,
                    training && cuda_graphs?pinned_losses:nullptr);graph.backward();
            }
            for(int layer=23;layer>=0;--layer) {
                arena.position=workspace;Graph graph(arena,blas,true);graph.context=context;graph.batch=sequences;
                int upstream=(23-layer)%2,downstream=1-upstream;
                cuda_check(cudaMemsetAsync(adj[downstream],0,n*4,cudaStreamPerThread));
                T x=graph.view(saved+layer*n,rows,Hidden,adj[downstream]);T out=layer_forward(graph,parameters,x,layer);
                out->grad=adj[upstream];graph.backward();
            }
        } else {
            Graph graph(arena, blas, training);graph.context=context;graph.batch=sequences;
            T hidden=forward(graph,parameters,ids,rows);
            metrics=loss(graph,parameters,hidden,target,targets,byte_lengths,accumulation,
                training && cuda_graphs?pinned_losses:nullptr);
            if(training)graph.backward();
        }
        if(capture) {
            cudaGraph_t captured;cuda_check(cudaStreamEndCapture(cudaStreamPerThread,&captured));
            cuda_check(cudaGraphInstantiate(&graph_exec,captured,0));cuda_check(cudaGraphDestroy(captured));
            cuda_check(cudaGraphLaunch(graph_exec,cudaStreamPerThread));
        }
        cuda_check(cudaStreamSynchronize(cudaStreamPerThread));
        if(training && cuda_graphs){metrics=collect_losses(pinned_losses,targets,byte_lengths,context);warmed=true;}
        return metrics;
    }
    Metrics evaluate(Tokens &data, int samples) {
        Metrics total;
        if(!evaluation_inputs.empty()) {
            for(auto &tokens:evaluation_inputs)total.add(batch(tokens,false));
            return total;
        }
        // Evenly spaced fixed blocks cover the chronological holdout. Repeated
        // evaluations use identical tokens, making loss trends interpretable.
        for (int i = 0; i < samples; ++i) {
            uint64_t start = uint64_t(i) * (data.count - context - 1) / std::max(samples - 1, 1);
            total.add(batch(data.block(start, context), false));
        }
        return total;
    }
};

template<class X> void write_binary(std::ostream &f, const X &x) { f.write(reinterpret_cast<const char *>(&x), sizeof(x)); }
template<class X> void read_binary(std::istream &f, X &x) { f.read(reinterpret_cast<char *>(&x), sizeof(x)); }
uint64_t checksum_bytes(uint64_t hash, const void *data, size_t bytes) {
    const auto *p=static_cast<const uint8_t *>(data);
    for(size_t i=0;i<bytes;++i) { hash^=p[i]; hash*=1099511628211ull; }
    return hash;
}
void durable_replace(const fs::path &temp, const fs::path &path) {
#ifdef _WIN32
    HANDLE handle=CreateFileW(temp.c_str(),GENERIC_READ|GENERIC_WRITE,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(handle==INVALID_HANDLE_VALUE)throw std::runtime_error("Cannot open checkpoint for durable flush");
    bool success=FlushFileBuffers(handle); CloseHandle(handle);
    if(!success)throw std::runtime_error("Cannot flush checkpoint to storage");
    if(!MoveFileExW(temp.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("Cannot atomically replace checkpoint");
#else
    int fd=open(temp.c_str(),O_RDWR);
    if(fd<0)throw std::runtime_error("Cannot open checkpoint for durable flush");
    int status=fsync(fd);close(fd);
    if(status)throw std::runtime_error("Cannot flush checkpoint to storage");
    fs::rename(temp,path);
    fd=open(path.parent_path().c_str(),O_RDONLY);
    if(fd>=0) { status=fsync(fd);close(fd);if(status)throw std::runtime_error("Cannot flush checkpoint directory"); }
#endif
}
int optimizer_id(const std::string &kind){return kind=="muon"?1:kind=="mantissa-adamw"?2:0;}
void checkpoint(const fs::path &path, Parameters &p, int step, int context, int accumulation, const std::mt19937 &rng,
                int physical_batch=1,const std::string &optimizer="adamw") {
    fs::create_directories(path.parent_path());
    fs::path temp = path; temp += ".tmp";
    std::ofstream file(temp, std::ios::binary);
    uint64_t magic = 0x5157454e4c4f5233ull;
    write_binary(file, magic); write_binary(file, p.count); write_binary(file, step);
    write_binary(file, context); write_binary(file, accumulation); write_binary(file, p.rank); write_binary(file, p.alpha);
    write_binary(file,physical_batch);int id=optimizer_id(optimizer);write_binary(file,id);
    std::ostringstream state; state << rng; std::string text = state.str(); uint64_t size = text.size();
    write_binary(file, size); file.write(text.data(), size);
    uint64_t hash=checksum_bytes(14695981039346656037ull,text.data(),text.size());
    std::vector<float> host(p.count);
    for (float *data : {p.values, p.moment1, p.moment2}) {
        cuda_check(cudaMemcpy(host.data(), data, p.count * 4, cudaMemcpyDeviceToHost));
        file.write(reinterpret_cast<const char *>(host.data()), p.count * 4);
        hash=checksum_bytes(hash,host.data(),p.count*4);
    }
    write_binary(file,hash);
    file.close(); if (!file) throw std::runtime_error("Checkpoint write failed");
    durable_replace(temp,path);
}
int restore(const fs::path &path, Parameters &p, int context, int accumulation, std::mt19937 &rng,
            int physical_batch=1,const std::string &optimizer="adamw") {
    std::ifstream file(path, std::ios::binary); uint64_t magic, count, size; int step, c, a, r; float alpha;
    read_binary(file, magic); read_binary(file, count); read_binary(file, step);
    read_binary(file, c); read_binary(file, a); read_binary(file, r); read_binary(file, alpha);
    int stored_batch=1,id=0;
    if(magic==0x5157454e4c4f5233ull){read_binary(file,stored_batch);read_binary(file,id);}
    read_binary(file,size);
    if (!file || (magic != 0x5157454e4c4f5232ull && magic!=0x5157454e4c4f5233ull) || count != p.count ||
        c != context || a<1 || stored_batch<1 || int64_t(a)*stored_batch != int64_t(accumulation)*physical_batch || r != p.rank || alpha != p.alpha ||
        id!=optimizer_id(optimizer) || size > 100000 || step < 0)
        throw std::runtime_error("Checkpoint format/configuration mismatch");
    std::string text(size, ' '); file.read(text.data(), size); std::istringstream state(text); state >> rng;
    if(!state)throw std::runtime_error("Invalid checkpoint RNG state");
    uint64_t hash=checksum_bytes(14695981039346656037ull,text.data(),text.size());
    std::vector<float> host(p.count);
    for (float *data : {p.values, p.moment1, p.moment2}) {
        file.read(reinterpret_cast<char *>(host.data()), p.count * 4);
        if (!file) throw std::runtime_error("Truncated checkpoint");
        hash=checksum_bytes(hash,host.data(),p.count*4);
        cuda_check(cudaMemcpy(data, host.data(), p.count * 4, cudaMemcpyHostToDevice));
    }
    uint64_t expected;read_binary(file,expected);
    if(!file || expected!=hash || file.peek()!=std::char_traits<char>::eof())
        throw std::runtime_error("Checkpoint checksum mismatch; state was not restored");
    return step;
}

double seconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
void log_metrics(std::ostream &file, TensorBoardLogger &logger, const Metrics &m, int step,
                 const std::string &kind, double elapsed, double memory, float lr = 0, float norm = 0, double update_seconds = 0,
                 int64_t processed_tokens=0,double training_seconds=0,uint64_t throughput_tokens=0,double allocated_gib=0) {
    std::ostringstream out; out << std::setprecision(9) << "{\"kind\":\"" << kind << "\",\"step\":" << step
        << ",\"loss\":" << m.loss() << ",\"bpb\":" << m.bpb() << ",\"last_loss\":" << m.last_loss()
        << ",\"last_bpb\":" << m.last_bpb() << ",\"predicted_tokens\":" << m.tokens
        << ",\"predicted_bytes\":" << m.bytes << ",\"last_tokens\":"<<m.last_tokens<<",\"last_bytes\":"<<m.last_bytes
        << ",\"last_nll\":"<<m.last_nll<<",\"processed_tokens\":"<<processed_tokens<<",\"seconds\":" << elapsed
        << ",\"peak_used_gib\":" << memory << ",\"learning_rate\":" << lr << ",\"grad_norm\":" << norm
        << ",\"update_seconds\":" << update_seconds << "}";
    file << out.str() << '\n'; file.flush(); std::cout << out.str() << std::endl;
    for (auto [tag, value] : std::vector<std::pair<std::string, double>>{
        {"loss", m.loss()}, {"bpb", m.bpb()}, {"last_loss", m.last_loss()}, {"last_bpb", m.last_bpb()},
        {"lr", lr}, {"grad_norm", norm}, {"peak_used_gib", memory}})
        logger.scalar(kind + "/" + tag, value, processed_tokens);
    if(kind=="train") {
        logger.scalar("loss",m.loss(),processed_tokens);
        for(size_t i=0;i<Metrics::Positions.size();++i)if(m.position_count[i])
            logger.scalar("loss_"+std::to_string(Metrics::Positions[i]),m.position_nll[i]/m.position_count[i],processed_tokens);
        logger.scalar("lr",lr,processed_tokens);
        logger.scalar("last_loss",m.last_loss(),processed_tokens);
        logger.scalar("bits-per-byte",m.last_bpb(),processed_tokens);
        logger.scalar("mem",allocated_gib,processed_tokens);
        logger.scalar("tokens_per_second",training_seconds>0?throughput_tokens/training_seconds:0,processed_tokens);
    }
    logger.flush();
}

void inference_export(Arena &arena, Blas &blas, Parameters &p, const std::string &input_path,
                      const fs::path &output, bool adapters, bool all_logits,const std::string &length_path) {
    std::ifstream input(input_path,std::ios::binary);
    uint64_t magic; uint32_t samples,context;
    read_binary(input,magic);read_binary(input,samples);read_binary(input,context);
    if(!input || magic!=0x514e494e50555431ull || samples<1 || context<1 || context>MaxContext)
        throw std::runtime_error("Invalid shared inference input");
    fs::create_directories(output);
    std::ofstream hidden_file(output/"hidden.f32",std::ios::binary),logits_file(output/"logits.f32",std::ios::binary),
        loss_file(output/"last_nll.f32",std::ios::binary);
    std::vector<uint32_t> lengths(Vocab);std::ifstream byte_file(length_path,std::ios::binary);
    byte_file.read(reinterpret_cast<char *>(lengths.data()),Vocab*4);if(!byte_file)throw std::runtime_error("Missing token byte lengths");
    Metrics metrics;
    for(uint32_t sample=0;sample<samples;++sample) {
        std::vector<uint32_t> tokens(context+1);input.read(reinterpret_cast<char *>(tokens.data()),tokens.size()*4);
        if(!input)throw std::runtime_error("Truncated inference input");
        for(auto id:tokens)if(id>=Vocab)throw std::runtime_error("Invalid inference token");
        arena.position=p.persistent;
        uint32_t *ids=arena.alloc<uint32_t>(context);cuda_check(cudaMemcpy(ids,tokens.data(),context*4,cudaMemcpyHostToDevice));
        Graph graph(arena,blas,false);graph.adapters=adapters;
        std::ofstream trace;
        if(sample==0 && std::getenv("QWEN_TRACE_LAYERS")) {
            trace.open(output/"first-sample-layers.f32",std::ios::binary);
            graph.observe_layer=[&](int layer,T value) {
                std::vector<float> host(value->size());cuda_check(cudaMemcpy(host.data(),value->data,host.size()*4,cudaMemcpyDeviceToHost));
                trace.write(reinterpret_cast<const char *>(host.data()),host.size()*4);
                if(!trace)throw std::runtime_error("Layer trace write failed");
            };
        }
        T h=forward(graph,p,ids,context);
        std::vector<float> hidden(h->size());cuda_check(cudaMemcpy(hidden.data(),h->data,hidden.size()*4,cudaMemcpyDeviceToHost));
        hidden_file.write(reinterpret_cast<const char *>(hidden.data()),hidden.size()*4);
        int start=all_logits?0:context-1;
        float *logits=arena.alloc<float>(size_t(64)*Vocab);
        float *last_loss=arena.alloc<float>(1);uint32_t *target=arena.alloc<uint32_t>(1);
        cuda_check(cudaMemcpy(target,&tokens.back(),4,cudaMemcpyHostToDevice));
        for(int row=start;row<int(context);row+=64) {
            int count=std::min(64,int(context)-row);
            p.head(blas,logits,h->data+row*Hidden,count);
            std::vector<float> host(size_t(count)*Vocab);cuda_check(cudaMemcpy(host.data(),logits,host.size()*4,cudaMemcpyDeviceToHost));
            logits_file.write(reinterpret_cast<const char *>(host.data()),host.size()*4);
            if(row+count==context) {
                cross_entropy<<<1,256>>>(logits+size_t(count-1)*Vocab,nullptr,target,last_loss,Vocab,0);
                float nll;cuda_check(cudaMemcpy(&nll,last_loss,4,cudaMemcpyDeviceToHost));
                loss_file.write(reinterpret_cast<const char *>(&nll),4);
                if(!std::isfinite(nll) || !lengths.at(tokens.back()))throw std::runtime_error("Invalid last-token metric");
                metrics.last_nll+=nll;++metrics.last_tokens;metrics.last_bytes+=lengths.at(tokens.back());
            }
        }
    }
    hidden_file.close();logits_file.close();loss_file.close();
    if(!hidden_file || !logits_file || !loss_file)throw std::runtime_error("Inference output write failed");
    std::ofstream report(output/"inference.json");
    report<<std::setprecision(12)<<"{\"samples\":"<<samples<<",\"context\":"<<context<<",\"vocab\":"<<Vocab
          <<",\"all_logits\":"<<(all_logits?"true":"false")<<",\"adapters\":"<<(adapters?"true":"false")
          <<",\"last_loss\":"<<metrics.last_loss()<<",\"last_bpb\":"<<metrics.last_bpb()<<",\"last_bytes\":"<<metrics.last_bytes
          <<",\"peak_used_gib\":"<<double(arena.peak)/GiB<<"}"<<std::endl;
    std::cout<<"{\"inference_export\":\"complete\",\"samples\":"<<samples<<"}"<<std::endl;
}
void tiled_delta_test(Arena &arena) {
    constexpr int t=70,heads=2,batch=2;size_t n=size_t(batch)*t*heads*128;
    size_t mark=arena.position;std::mt19937 rng(17);std::uniform_real_distribution<float> random(-0.1f,0.1f);
    float *inputs[5],*old_grad[5],*new_grad[5];
    for(int i=0;i<5;++i) {
        size_t count=i<3?n:n/128;std::vector<float> host(count);
        for(float &x:host)x=i==3?0.7f+random(rng):i==4?0.5f+random(rng):random(rng);
        inputs[i]=arena.alloc<float>(count);old_grad[i]=arena.alloc<float>(count);new_grad[i]=arena.alloc<float>(count);
        cuda_check(cudaMemcpy(inputs[i],host.data(),count*4,cudaMemcpyHostToDevice));
        cuda_check(cudaMemset(old_grad[i],0,count*4));cuda_check(cudaMemset(new_grad[i],0,count*4));
    }
    float *old_out=arena.alloc<float>(n),*new_out=arena.alloc<float>(n),*dy=arena.alloc<float>(n);
    std::vector<float> objective(n);for(float &x:objective)x=random(rng);
    cuda_check(cudaMemcpy(dy,objective.data(),n*4,cudaMemcpyHostToDevice));
    float *old_states=arena.alloc<float>(n*128);
    float *checkpoints=arena.alloc<float>(size_t(batch)*((t+DeltaChunk-1)/DeltaChunk)*heads*128*128);
    for(int seq=0;seq<batch;++seq) {
        size_t offset=size_t(seq)*t*heads*128,scalar_offset=offset/128;
        delta_forward<<<heads,128>>>(inputs[0]+offset,inputs[1]+offset,inputs[2]+offset,inputs[3]+scalar_offset,inputs[4]+scalar_offset,
            old_out+offset,old_states+offset*128,t,heads);
        delta_backward<<<heads,128>>>(inputs[0]+offset,inputs[1]+offset,inputs[2]+offset,inputs[3]+scalar_offset,inputs[4]+scalar_offset,
            old_states+offset*128,dy+offset,old_grad[0]+offset,old_grad[1]+offset,old_grad[2]+offset,
            old_grad[3]+scalar_offset,old_grad[4]+scalar_offset,t,heads);
    }
    delta_tiled_forward<<<batch*heads*8,128>>>(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],new_out,checkpoints,t,heads);
    float *history=nullptr,*errors=arena.alloc<float>(size_t(batch)*DeltaChunk*heads*128);
    float2 *qk=arena.alloc<float2>(n*DeltaGroups),*db=arena.alloc<float2>(n/128*DeltaGroups);
    delta_tiled_backward<<<batch*heads*8,128>>>(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],checkpoints,dy,
        new_grad[2],history,errors,qk,db,t,heads);
    delta_reduce<<<blocks(n),256>>>(qk,db,new_grad[0],new_grad[1],new_grad[3],new_grad[4],t,heads,batch);
    double maximum=0;
    for(int i=-1;i<5;++i) {
        size_t count=i<3?n:n/128;std::vector<float> old(count),now(count);
        cuda_check(cudaMemcpy(old.data(),i<0?old_out:old_grad[i],count*4,cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(now.data(),i<0?new_out:new_grad[i],count*4,cudaMemcpyDeviceToHost));
        for(size_t e=0;e<count;++e) {
            double error=std::abs(old[e]-now[e]);maximum=std::max(maximum,error);
            if(error>2e-7+3e-5*std::abs(old[e]))throw std::runtime_error("Tiled/batched DeltaNet forward or backward mismatch");
        }
    }
    arena.position=mark;
    std::cout<<"{\"gradient_check\":\"tiled-batched-DeltaNet-multiple-checkpoint-chunks\",\"max_abs\":"<<maximum<<",\"status\":\"passed\"}"<<std::endl;
}
int self_test();
void optimizer_probe(Parameters &p,Optimizer &optimizer,const fs::path &directory) {
    fs::create_directories(directory);std::mt19937 rng(192);std::normal_distribution<float> normal(0,0.01f);
    std::vector<float> host(p.count);
    auto save=[&](const std::string &name,float *data) {
        cuda_check(cudaMemcpy(host.data(),data,p.count*4,cudaMemcpyDeviceToHost));
        std::ofstream file(directory/name,std::ios::binary);file.write(reinterpret_cast<const char *>(host.data()),p.count*4);
        if(!file)throw std::runtime_error("Optimizer probe output failed");
    };
    save("initial.f32",p.values);
    for(float &value:host)value=normal(rng);
    cuda_check(cudaMemcpy(p.gradients,host.data(),p.count*4,cudaMemcpyHostToDevice));save("gradient.f32",p.gradients);
    for(int step=1;step<=20;++step)optimizer.step(step,optimizer.kind=="muon"?0.001f:0.0001f,1);
    save("values.f32",p.values);save("momentum.f32",p.moment1);save("variance.f32",p.moment2);
    std::ofstream file(directory/"shapes.json");file<<"[";bool first=true;
    for(auto &[name,l]:p.linears) {
        for(auto m:{original_optimizer::Matrix{int(l.a-p.values),l.rank,l.input,0},
                    original_optimizer::Matrix{int(l.b-p.values),l.output,l.rank,0}}) {
            if(!first)file<<",";first=false;file<<"{\"offset\":"<<m.weight<<",\"rows\":"<<m.rows<<",\"cols\":"<<m.cols<<"}";
        }
    }
    file<<"]\n";
    std::cout<<"{\"optimizer_probe\":\"complete\",\"optimizer\":\""<<optimizer.kind<<"\"}"<<std::endl;
}
int main(int argc, char **argv) {
    try {
        // Match nvidia-smi's physical ordering, including on mixed GPU systems.
#ifdef _WIN32
        _putenv_s("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
#else
        setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 1);
#endif
        int device = experiment::Device, context = experiment::Context, accumulation = 1, physical_batch=-1, total_batch=ReferenceBatch, steps = experiment::Steps, rank = experiment::Rank, eval_every = experiment::EvalEvery, eval_samples = experiment::EvalSamples, warmup_steps = experiment::WarmupSteps,profile_step=0,workspace_mib=1024,benchmark_warmup=2;
        double budget = -1, duration = 0; float learning_rate = 0, alpha = experiment::Alpha;
        uint32_t seed = experiment::Seed; bool test = false, adapters = true, all_logits = false, accumulation_set=false;
        std::string assets = experiment::Assets, output = experiment::Output, resume, inference_input, train_input, precision=experiment::Precision;
        std::string optimizer_kind=experiment::Optimizer,schedule=experiment::Schedule,checkpointing="layer",eval_input,probe,graphs="on";
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--self-test") { test = true; continue; }
            if (key == "--no-lora") { adapters=false;continue; }
            if (key == "--all-logits") { all_logits=true;continue; }
            if (key == "--help") {
                std::cout << "Native CUDA C++ Qwen3.5-0.8B LoRA trainer\n"
                    "--device 1 --context 512 --batch 16 --steps 100 --seconds 0\n"
                    "--context 8..4096; native/experiment.h sets the experiment's effective batch\n"
                    "Other contexts use effective sequences = 65536 / max(context,512), rounded down\n"
                    "--batch N (alias --virtual-batch): physical sequences; must divide effective batch\n"
                    "Default batch scales from 16 at context 512 to 2 at 4096; accumulation is automatic.\n"
                    "--rank 16 --alpha 32 --optimizer muon|mantissa-adamw|adamw --lr 0.001 --warmup-steps 3\n"
                    "--schedule warmup|onecycle (original TurboGPT OneCycle implementation)\n"
                    "--checkpointing layer|off\n"
                    "--eval-input PATH (the validator's exact shared random subset)\n"
                    "--train-input PATH (shared benchmark windows; same binary format as --eval-input)\n"
                    "--benchmark-warmup 2 (updates excluded only from steady-state throughput)\n"
                    "--profile-step N --optimizer-probe PATH\n"
                    "--cuda-graphs on|off\n"
                    "--blas-workspace-mib 1024 (inside the arena)\n"
                    "--eval-every 10 --eval-samples 16 --vram-gib 12 --seed 3407\n"
                    "--assets .cache/native/paths.txt --output runs/native-qwen\n"
                    "--resume PATH --self-test\n--inference-input PATH --no-lora --all-logits --precision tf32x2|fp32|tf32|bf16|bf16x9\n";
                return 0;
            }
            if (i + 1 == argc) throw std::runtime_error("Missing option value for " + key);
            std::string v = argv[++i];
            if (key == "--device") device = std::stoi(v); else if (key == "--context") context = std::stoi(v);
            else if (key == "--accum") {accumulation = std::stoi(v);accumulation_set=true;} else if (key == "--steps") steps = std::stoi(v);
            else if(key=="--batch" || key=="--virtual-batch")physical_batch=std::stoi(v);
            else if (key == "--rank") rank = std::stoi(v); else if (key == "--alpha") alpha = std::stof(v);
            else if (key == "--warmup-steps") warmup_steps = std::stoi(v);
            else if (key == "--eval-every") eval_every = std::stoi(v); else if (key == "--eval-samples") eval_samples = std::stoi(v);
            else if (key == "--vram-gib") budget = std::stod(v); else if (key == "--seconds") duration = std::stod(v);
            else if (key == "--lr") learning_rate = std::stof(v); else if (key == "--seed") seed = uint32_t(std::stoul(v));
            else if (key == "--assets") assets = v; else if (key == "--output") output = v; else if (key == "--resume") resume = v;
            else if (key == "--inference-input") inference_input=v; else if (key == "--precision") precision=v;
            else if(key=="--optimizer")optimizer_kind=v;else if(key=="--schedule")schedule=v;
            else if(key=="--checkpointing")checkpointing=v;
            else if(key=="--eval-input")eval_input=v;
            else if(key=="--train-input")train_input=v;
            else if(key=="--benchmark-warmup")benchmark_warmup=std::stoi(v);
            else if(key=="--profile-step")profile_step=std::stoi(v);else if(key=="--optimizer-probe")probe=v;
            else if(key=="--cuda-graphs")graphs=v;
            else if(key=="--blas-workspace-mib")workspace_mib=std::stoi(v);
            else throw std::runtime_error("Unknown option " + key);
        }
        if(context<8 || context>MaxContext)throw std::runtime_error("Context range is 8..4096");
        total_batch=context==experiment::Context?experiment::EffectiveBatch:effective_batch(context);
        if(physical_batch==-1)physical_batch=context==experiment::Context?experiment::Batch:default_batch(context,total_batch);
        if(budget==-1)budget=context==experiment::Context?experiment::VramGiB:12;
        if(physical_batch<1 || accumulation<1 || total_batch<1 || benchmark_warmup<0)
            throw std::runtime_error("Batch sizes and accumulation must be positive; benchmark warmup must be nonnegative");
        // Retain explicit --accum only for old checkpoints and tiny validation runs.
        if(accumulation_set) {
            int64_t product=int64_t(physical_batch)*accumulation;
            if(product>INT_MAX)throw std::runtime_error("Total batch size is too large");
            total_batch=int(product);
        } else {
            if(total_batch%physical_batch)throw std::runtime_error("--batch must divide the effective batch of "+std::to_string(total_batch)+" exactly");
            int inferred=total_batch/physical_batch;
            accumulation=inferred;
        }
        cuda_check(cudaSetDevice(device));
        if (test) return self_test();
        if(learning_rate==0)learning_rate=optimizer_kind==experiment::Optimizer?experiment::LearningRate:(optimizer_kind=="muon"?0.001f:0.0001f);
        if(schedule!="warmup" && schedule!="onecycle")throw std::runtime_error("Unknown schedule");
        if (accumulation < 1 || physical_batch<1 || workspace_mib<1 || steps < 1 || rank < 1 || eval_every < 1 || eval_samples < 1 || budget <= 0 || warmup_steps < 0 || learning_rate <= 0 || alpha <= 0 || duration < 0)
            throw std::runtime_error("Invalid trainer configuration");
        if (resume.empty() && probe.empty() && profile_step==0)
            for(const char *name:{"metrics.jsonl","checkpoint.bin","report.json","hidden.f32","logits.f32"})
                if(fs::exists(fs::path(output)/name))throw std::runtime_error("Output contains training/inference results; choose a new output or --resume");
        std::ifstream paths(assets); std::string model, manifest, train_path, eval_path, lengths;
        std::getline(paths, model); std::getline(paths, manifest); std::getline(paths, train_path);
        std::getline(paths, eval_path); std::getline(paths, lengths);
        if (!paths) throw std::runtime_error("Run tools/native_assets.py first");
        for (std::string *p : {&model, &manifest, &train_path, &eval_path, &lengths}) if (!p->empty() && p->back() == '\r') p->pop_back();
        cudaDeviceProp prop; cuda_check(cudaGetDeviceProperties(&prop, device));
        size_t free, total; cuda_check(cudaMemGetInfo(&free, &total));
        size_t capacity = size_t(budget * GiB);
        if (capacity + GiB / 2 > free) throw std::runtime_error("Insufficient free VRAM for requested arena plus driver headroom");
        std::cout << "{\"gpu\":\"" << prop.name << "\",\"device\":" << device << ",\"arena_gib\":" << budget
                  << ",\"context\":" << context << ",\"batch\":"<<physical_batch<<",\"total_batch\":"<<total_batch
                  <<",\"tokens_per_update\":"<<int64_t(total_batch)*context<<",\"accumulation\":" << accumulation
                  <<",\"sampling\":\""<<(experiment::Sequential?"sequential":"random")<<"\",\"evaluation_enabled\":"<<(experiment::Evaluate?"true":"false")<<",\"pytorch\":false}" << std::endl;
        Arena arena(capacity); Blas blas;
        if(precision=="fp32")blas.compute=CUBLAS_COMPUTE_32F;
        else if(precision=="tf32x2")blas.corrected_tf32=true;
        else if(precision=="bf16")blas.native_bf16=true;
        else if(precision=="bf16x9")blas.compute=CUBLAS_COMPUTE_32F_EMULATED_16BFX9;
        else if(precision!="tf32")throw std::runtime_error("Unknown precision");
        Parameters parameters(arena, model, manifest, rank, alpha, seed,blas.native_bf16);
        blas.scratch_arena=&arena;
        for(auto &[name,w]:parameters.weights)if(w.bf16 && w.data)blas.exact_tf32.insert(w.data);
        blas.workspace(arena,size_t(workspace_mib)*1024*1024);parameters.persistent=arena.position;
        if(!inference_input.empty()) {
            if(!resume.empty())throw std::runtime_error("Inference validation expects the original model, without a resumed adapter");
            inference_export(arena,blas,parameters,inference_input,output,adapters,all_logits,lengths);return 0;
        }
        Optimizer optimizer(arena,blas,parameters,optimizer_kind);
        if(!probe.empty()){optimizer_probe(parameters,optimizer,probe);return 0;}
        Runner runner(arena, blas, parameters, lengths, context, accumulation);
        if(checkpointing!="layer" && checkpointing!="off")throw std::runtime_error("Unknown checkpointing policy");
        runner.checkpoint_layers=checkpointing=="layer";
        if(graphs!="on" && graphs!="off")throw std::runtime_error("Unknown CUDA graph policy");
        runner.cuda_graphs=graphs=="on";
        if(!eval_input.empty()) {
            std::ifstream file(eval_input,std::ios::binary);uint64_t magic;uint32_t samples,c;
            read_binary(file,magic);read_binary(file,samples);read_binary(file,c);
            if(!file || magic!=0x514e494e50555431ull || c!=context || samples<1 || samples>100000)
                throw std::runtime_error("Invalid shared evaluation inputs");
            eval_samples=int(samples);runner.evaluation_inputs.resize(samples,std::vector<uint32_t>(context+1));
            for(auto &tokens:runner.evaluation_inputs) {
                file.read(reinterpret_cast<char *>(tokens.data()),tokens.size()*4);
                if(!file)throw std::runtime_error("Truncated shared evaluation inputs");
                for(auto id:tokens)if(id>=Vocab)throw std::runtime_error("Invalid shared evaluation token");
            }
            if(file.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Trailing shared evaluation input data");
        }
        Tokens training(train_path), validation(eval_path);
        std::vector<uint32_t> shared_training;uint32_t shared_samples=0;
        if(!train_input.empty()) {
            std::ifstream file(train_input,std::ios::binary);uint64_t magic;uint32_t c;
            read_binary(file,magic);read_binary(file,shared_samples);read_binary(file,c);
            if(!file || magic!=0x514e494e50555431ull || c!=context || shared_samples<uint32_t(total_batch) || shared_samples>100000)
                throw std::runtime_error("Invalid shared training inputs; provide at least one full optimizer batch of windows");
            shared_training.resize(size_t(shared_samples)*(context+1));
            file.read(reinterpret_cast<char *>(shared_training.data()),shared_training.size()*4);
            if(!file || file.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Truncated/trailing shared training input data");
            for(auto id:shared_training)if(id>=Vocab)throw std::runtime_error("Invalid shared training token");
        }
        if (training.count <= uint64_t(context) || (experiment::Evaluate && validation.count <= uint64_t(context)))
            throw std::runtime_error("Dataset is shorter than one context");
        std::vector<uint64_t> sequential_starts;
        if constexpr(experiment::Sequential) {
            if(!shared_training.empty())throw std::runtime_error("Sequential sampling cannot use shared training windows");
            std::ifstream positions(experiment::SequentialStarts,std::ios::binary);
            sequential_starts.resize(total_batch);
            positions.read(reinterpret_cast<char *>(sequential_starts.data()),sequential_starts.size()*8);
            if(!positions || positions.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Sequential starts must contain exactly one uint64 per effective batch lane");
            for(int j=0;j<total_batch;++j) {
                uint64_t boundary=j+1<total_batch?sequential_starts[j+1]:training.count;
                uint64_t span=uint64_t(steps)*context+1;
                if(sequential_starts[j]>=boundary || span>boundary-sequential_starts[j])
                    throw std::runtime_error("Sequential lane "+std::to_string(j)+" crosses the next starting point");
            }
        }
        std::mt19937 rng(seed); int completed = resume.empty() ? 0 : restore(resume, parameters, context, accumulation, rng,physical_batch,optimizer_kind);
        optimizer.sync_bits();
        fs::create_directories(output); TensorBoardLogger logger(output);
        {
            std::ofstream sampler(fs::path(output)/"sampler.json");
            sampler<<"{\"sampling\":\""<<(experiment::Sequential?"sequential":"random")<<"\",\"corpus_tokens\":"<<training.count
                <<",\"context\":"<<context<<",\"effective_batch\":"<<total_batch<<",\"steps\":"<<steps
                <<",\"restored_step\":"<<completed<<",\"next_lane_advance_tokens\":"<<uint64_t(completed)*context<<",\"starts\":[";
            for(size_t j=0;j<sequential_starts.size();++j){if(j)sampler<<",";sampler<<sequential_starts[j];}
            sampler<<"]}\n";
        }
        uint64_t model_bytes=parameters.count*sizeof(float);
        for(const auto &[name,w]:parameters.weights)model_bytes+=w.count*(w.packed?2:4);
        logger.scalar("model_bytes",double(model_bytes),0);
        logger.scalar("data_bytes",double(experiment::DataBytes),0);
        logger.scalar("train_batch_size",physical_batch,0);
        logger.scalar("train_effective_batch_size",total_batch,0);
        logger.text("revision",build_revision(),0);
        logger.text("model_size","Qwen3.5-0.8B-Base; 24 text layers, width 1024; LoRA parameters "+std::to_string(parameters.count),0);
        logger.text("dataset",experiment::Dataset,0);
        logger.text("data_size",std::to_string(experiment::DataBytes)+" bytes",0);
        const char *node=std::getenv("COMPUTERNAME");if(!node)node=std::getenv("HOSTNAME");
        logger.text("node",std::string(node?node:"unknown")+"; "+prop.name+"; device "+std::to_string(device),0);
        logger.text("optimizer",optimizer_kind+"; lr="+std::to_string(learning_rate)+"; "+schedule+"; warmup="+std::to_string(warmup_steps)+"; clip=1",0);
        logger.text("description",std::string("Native CUDA C++ LoRA; context=")+std::to_string(context)+"; rank="+std::to_string(rank)+"; alpha="+std::to_string(alpha)+"; precision="+precision+"; sampler="+(experiment::Sequential?"sequential":"random")+"; evaluation="+(experiment::Evaluate?"enabled":"disabled"),0);
        logger.flush();
        if (fs::exists(fs::path(assets).parent_path() / "assets.json"))
            fs::copy_file(fs::path(assets).parent_path() / "assets.json", fs::path(output) / "assets.json", fs::copy_options::overwrite_existing);
        {
            std::ofstream command(fs::path(output) / (resume.empty() ? "command.txt" : "resume-command.txt"));
            for (int i = 0; i < argc; ++i) command << '"' << argv[i] << "\" ";
            command << '\n';
        }
        std::ofstream history(fs::path(output) / "metrics.jsonl", resume.empty() ? std::ios::trunc : std::ios::app);
        auto start = std::chrono::steady_clock::now();
        Metrics baseline,final;
        if constexpr(experiment::Evaluate) {
            baseline=runner.evaluate(validation,eval_samples);
            log_metrics(history,logger,baseline,completed,"eval",seconds(start),double(arena.peak)/GiB,0,0,0,int64_t(completed)*total_batch*context);
        }
        std::uniform_int_distribution<uint64_t> sample(0, training.count - context - 1);
        Metrics total_training; double training_seconds = 0,steady_seconds=0;uint64_t steady_tokens=0;int run_updates=0;
        auto last_checkpoint=std::chrono::steady_clock::now();
        for (int step = completed + 1; step <= steps; ++step) {
            if(profile_step==step && cuProfilerStart()!=CUDA_SUCCESS)throw std::runtime_error("Profiler start failed");
            auto batch_start = std::chrono::steady_clock::now();
            cuda_check(cudaMemset(parameters.gradients, 0, parameters.count * 4));
            Metrics batch;
            for (int micro = 0; micro < accumulation; ++micro) {
                std::vector<uint32_t> tokens;
                tokens.reserve(size_t(physical_batch)*(context+1));
                for(int b=0;b<physical_batch;++b) {
                    if(shared_training.empty()) {
                        uint64_t position;
                        if constexpr(experiment::Sequential)position=sequential_starts[micro*physical_batch+b]+uint64_t(step-1)*context;
                        else position=sample(rng);
                        auto block=training.block(position,context);tokens.insert(tokens.end(),block.begin(),block.end());
                    } else {
                        size_t index=(uint64_t(step-1)*total_batch+micro*physical_batch+b)%shared_samples;
                        auto begin=shared_training.begin()+index*(context+1);tokens.insert(tokens.end(),begin,begin+context+1);
                    }
                }
                batch.add(runner.batch(tokens,true));
            }
            arena.position = parameters.persistent;
            float *sum = arena.alloc<float>(1); cuda_check(cudaMemset(sum, 0, 4));
            gradient_norm<<<1024, 256>>>(parameters.gradients, sum, parameters.count);
            float norm2; cuda_check(cudaMemcpy(&norm2, sum, 4, cudaMemcpyDeviceToHost));
            if (!std::isfinite(norm2)) throw std::runtime_error("Non-finite adapter gradient norm");
            float norm = std::sqrt(norm2), clip = std::min(1.f, 1.f / std::max(norm, 1e-12f));
            float warmup = std::min(1.f, float(step) / std::max(1, warmup_steps));
            float lr = learning_rate * warmup; // Constant after warmup; suitable for time-limited exploration/resume.
            float momentum=0.95f;
            if(schedule=="onecycle") {
                OneCycle cycle;cycle.total_steps=steps;auto point=onecycle_point(cycle,step-1);
                lr=learning_rate*point.multiplier;momentum=point.momentum;
            }
            optimizer.step(step,lr,clip,momentum);
            cuda_check(cudaDeviceSynchronize()); completed = step; total_training.add(batch);
            if(profile_step==step && cuProfilerStop()!=CUDA_SUCCESS)throw std::runtime_error("Profiler stop failed");
            double update_seconds=seconds(batch_start);training_seconds+=update_seconds;++run_updates;
            if(run_updates>benchmark_warmup){steady_seconds+=update_seconds;steady_tokens+=batch.tokens;}
            int64_t processed_tokens=int64_t(step)*total_batch*context;
            log_metrics(history, logger, batch, step, "train", seconds(start), double(arena.peak) / GiB, lr, norm,update_seconds,processed_tokens,training_seconds,total_training.tokens,double(arena.capacity)/GiB);
            if (experiment::Evaluate && step % eval_every == 0) {
                Metrics evaluation = runner.evaluate(validation, eval_samples);
                log_metrics(history, logger, evaluation, step, "eval", seconds(start), double(arena.peak) / GiB,0,0,0,processed_tokens);
            }
            if ((experiment::Evaluate && step % eval_every == 0) || seconds(last_checkpoint)>=experiment::CheckpointSeconds) {
                checkpoint(fs::path(output) / "checkpoint.bin", parameters, completed, context, accumulation, rng,physical_batch,optimizer_kind);
                last_checkpoint=std::chrono::steady_clock::now();
            }
            if (duration > 0 && seconds(start) >= duration) break;
        }
        if constexpr(experiment::Evaluate) {
            final=runner.evaluate(validation,eval_samples);
            log_metrics(history,logger,final,completed,"eval",seconds(start),double(arena.peak)/GiB,0,0,0,int64_t(completed)*total_batch*context);
        }
        checkpoint(fs::path(output) / "checkpoint.bin", parameters, completed, context, accumulation, rng,physical_batch,optimizer_kind);
        std::ofstream report(fs::path(output) / "report.json");
        auto eval_value=[](double value) {
            if(!experiment::Evaluate)return std::string("null");
            std::ostringstream text;text<<std::setprecision(10)<<value;return text.str();
        };
        report << std::setprecision(10) << "{\"steps\":" << completed << ",\"context\":" << context
            << ",\"batch\":"<<physical_batch<<",\"total_batch\":"<<total_batch<<",\"tokens_per_update\":"<<int64_t(total_batch)*context
            <<",\"optimizer\":\""<<optimizer_kind<<"\",\"schedule\":\""<<schedule
            <<"\",\"checkpointing\":\""<<checkpointing<<"\",\"cuda_graphs\":\""<<graphs<<"\",\"accumulation\":" << accumulation << ",\"eval_samples\":" << eval_samples
            << ",\"rank\":" << rank << ",\"alpha\":" << alpha << ",\"learning_rate\":" << learning_rate
            << ",\"precision\":\""<<precision<<"\",\"blas_workspace_mib\":"<<workspace_mib<<",\"gpu\":\"" << prop.name << "\",\"device\":" << device
            << ",\"training_seconds\":" << training_seconds << ",\"process_seconds\":" << seconds(start)
            << ",\"training_tokens\":" << total_training.tokens << ",\"tokens_per_second\":" << total_training.tokens / std::max(training_seconds, 1e-9)
            << ",\"benchmark_warmup_updates\":"<<benchmark_warmup<<",\"steady_training_seconds\":"<<steady_seconds
            << ",\"steady_training_tokens\":"<<steady_tokens<<",\"steady_tokens_per_second\":"<<steady_tokens/std::max(steady_seconds,1e-9)
            << ",\"processed_tokens\":"<<int64_t(completed)*total_batch*context
            << ",\"evaluation_enabled\":"<<(experiment::Evaluate?"true":"false")<<",\"sampling\":\""<<(experiment::Sequential?"sequential":"random")<<"\""
            << ",\"baseline_eval_loss\":" << eval_value(baseline.loss()) << ",\"final_eval_loss\":" << eval_value(final.loss())
            << ",\"baseline_eval_bpb\":" << eval_value(baseline.bpb()) << ",\"final_eval_bpb\":" << eval_value(final.bpb())
            << ",\"baseline_eval_last_loss\":" << eval_value(baseline.last_loss()) << ",\"final_eval_last_loss\":" << eval_value(final.last_loss())
            << ",\"baseline_eval_last_bpb\":"<<eval_value(baseline.last_bpb())<<",\"final_eval_last_bpb\":" << eval_value(final.last_bpb()) << ",\"peak_used_gib\":" << double(arena.peak) / GiB
            << ",\"reserved_gib\":" << budget << ",\"pytorch\":false}" << std::endl;
        return 0;
    } catch (const std::exception &e) { std::cerr << "error: " << e.what() << std::endl; return 1; }
}

// Native finite-difference checks are defined separately below.
double graph_tests(Arena &arena, Blas &blas) {
    blas.compute = CUBLAS_COMPUTE_32F;
    std::mt19937 rng(123); std::uniform_real_distribution<float> random(-0.3f, 0.3f);
    auto buffer = [&](int size) {
        float *gpu = arena.alloc<float>(size); std::vector<float> host(size);
        for (float &x : host) x = random(rng);
        cuda_check(cudaMemcpy(gpu, host.data(), size * 4, cudaMemcpyHostToDevice));
        return gpu;
    };
    auto get = [&](float *gpu, size_t size) {
        std::vector<float> host(size); cuda_check(cudaMemcpy(host.data(), gpu, size * 4, cudaMemcpyDeviceToHost)); return host;
    };
    float *wnorm = buffer(128), *wconv = buffer(512 * 4), *alog = buffer(16), *bias = buffer(16);
    Linear linear{buffer(5 * 7), buffer(3 * 7), buffer(3 * 5), arena.alloc<float>(3 * 7), arena.alloc<float>(3 * 5), 7, 5, 3, 2};
    size_t permanent = arena.position; double maximum = 0;
    // Exercise composition, accumulation through residuals and group slicing,
    // plus both forms of normalization and attention/MLP output gating.
    using Operation = std::function<T(Graph &, const std::vector<T> &)>;
    auto check = [&](const std::string &name, std::vector<std::pair<int,int>> shapes, Operation operation) {
        std::vector<std::vector<float>> host(shapes.size()), analytic(shapes.size());
        for (size_t i = 0; i < host.size(); ++i) {
            host[i].resize(shapes[i].first * shapes[i].second);
            for (float &v : host[i]) v = random(rng);
        }
        std::vector<float> objective;
        auto evaluate = [&](bool training) {
            arena.position = permanent; Graph graph(arena, blas, training); std::vector<T> input;
            for (size_t i = 0; i < host.size(); ++i) {
                T x = graph.tensor(shapes[i].first, shapes[i].second);
                cuda_check(cudaMemcpy(x->data, host[i].data(), x->size() * 4, cudaMemcpyHostToDevice)); input.push_back(x);
            }
            T out = operation(graph, input);
            if (objective.empty()) { objective.resize(out->size()); for (float &v : objective) v = random(rng); }
            std::vector<float> y = get(out->data, out->size());
            if (name == "grouped-causal-attention") {
                for (int h = 0; h < 8; ++h) for (int row = 0; row < 4; ++row) {
                    double score[4], total = 0, high = -INFINITY;
                    for (int j = 0; j <= row; ++j) {
                        score[j] = 0;
                        for (int d = 0; d < 256; ++d)
                            score[j] += double(host[0][row*2048+h*256+d]) * host[1][j*512+(h/4)*256+d] / 16;
                        high = std::max(high, score[j]);
                    }
                    for (int j = 0; j <= row; ++j) { score[j] = std::exp(score[j]-high); total += score[j]; }
                    for (int d = 0; d < 256; ++d) {
                        double expected = 0;
                        for (int j = 0; j <= row; ++j) expected += score[j] / total * host[2][j*512+(h/4)*256+d];
                        if (std::abs(expected-y[row*2048+h*256+d]) > 3e-6)
                            throw std::runtime_error("Attention CPU forward reference mismatch");
                    }
                }
            }
            if (training) {
                cuda_check(cudaMemcpy(graph.grad(out), objective.data(), out->size() * 4, cudaMemcpyHostToDevice)); graph.backward();
                for (size_t i = 0; i < host.size(); ++i) analytic[i] = get(graph.grad(input[i]), input[i]->size());
            }
            return std::inner_product(y.begin(), y.end(), objective.begin(), 0.0);
        };
        evaluate(true);
        for (size_t i = 0; i < host.size(); ++i) {
            // Deterministic positions span sequence ends, head boundaries and interiors.
            for (size_t index : {size_t(0), size_t(7), host[i].size()/3, host[i].size()/2, host[i].size()-1}) {
                if (index >= host[i].size()) continue;
                float original = host[i][index], epsilon = 0.002f;
                host[i][index] = original + epsilon; double plus = evaluate(false);
                host[i][index] = original - epsilon; double minus = evaluate(false);
                host[i][index] = original;
                double expected = (plus - minus) / (2 * epsilon), error = std::abs(expected - analytic[i][index]);
                maximum = std::max(maximum, error);
                if (error > 2e-4 + 0.005 * std::abs(expected)) throw std::runtime_error(name + " finite-difference mismatch");
            }
        }
        std::cout << "{\"gradient_check\":\"" << name << "\",\"status\":\"passed\"}" << std::endl;
    };
    check("RMSNorm-convolution-RoPE-SiLU-residual", {{4,512},{4,512}}, [&](Graph &g, const std::vector<T> &x) {
        return g.add(x[0], g.gated(g.rope(g.conv(g.norm(x[0],128,wnorm,1),wconv),2),x[1],true));
    });
    check("L2Norm-sigmoid-gating", {{4,512},{4,512}}, [&](Graph &g, const std::vector<T> &x) {
        return g.gated(g.norm(x[0],128,nullptr,0,true),x[1],false);
    });
    check("decay-sigmoid", {{4,16}}, [&](Graph &g, const std::vector<T> &x) { return g.add(g.decay(x[0],alog,bias),g.sig(x[0])); });
    check("grouped-causal-attention", {{4,2048},{4,512},{4,512}}, [&](Graph &g, const std::vector<T> &x) { return g.attention(x[0],x[1],x[2]); });
    check("query-gate-slice", {{4,4096}}, [&](Graph &g, const std::vector<T> &x) {
        return g.gated(g.slice(x[0],8,256,512,256),g.slice(x[0],8,256,512,0),false);
    });
    cuda_check(cudaMemset(linear.ga,0,3*7*4)); cuda_check(cudaMemset(linear.gb,0,3*5*4));
    check("LoRA-input-gradient", {{4,7}}, [&](Graph &g, const std::vector<T> &x) { return g.linear(x[0],linear); });
    // Check adapter derivatives independently against a CPU scalar objective.
    // All three GEMM transpose paths must be correct for LoRA to train.
    arena.position = permanent; Graph graph(arena,blas,true); T input = graph.tensor(4,7);
    std::vector<float> hx(28), objective(20); for(float &v:hx)v=random(rng); for(float &v:objective)v=random(rng);
    cuda_check(cudaMemcpy(input->data,hx.data(),28*4,cudaMemcpyHostToDevice));
    T out = graph.linear(input,linear);
    cuda_check(cudaMemset(linear.ga,0,3*7*4)); cuda_check(cudaMemset(linear.gb,0,3*5*4));
    cuda_check(cudaMemcpy(graph.grad(out),objective.data(),20*4,cudaMemcpyHostToDevice)); graph.backward();
    auto ha=get(linear.a,21), hb=get(linear.b,15), hw=get(linear.weight,35), ga=get(linear.ga,21), gb=get(linear.gb,15);
    auto cpu = [&] {
        double value=0;
        for(int t=0;t<4;++t)for(int o=0;o<5;++o) {
            double y=0; for(int i=0;i<7;++i) y+=hx[t*7+i]*hw[o*7+i];
            for(int r=0;r<3;++r) { double projected=0; for(int i=0;i<7;++i) projected+=hx[t*7+i]*ha[r*7+i]; y+=2*projected*hb[o*3+r]; }
            value+=y*objective[t*5+o];
        } return value;
    };
    auto hy=get(out->data,20);
    if(std::abs(cpu()-std::inner_product(hy.begin(),hy.end(),objective.begin(),0.0))>2e-6)
        throw std::runtime_error("LoRA CPU forward reference mismatch");
    for(auto pair:{std::make_pair(&ha,&ga),std::make_pair(&hb,&gb)})for(size_t i=0;i<pair.first->size();++i) {
        float original=pair.first->at(i),epsilon=0.002f;
        pair.first->at(i)=original+epsilon; double plus=cpu(); pair.first->at(i)=original-epsilon; double minus=cpu(); pair.first->at(i)=original;
        double error=std::abs((plus-minus)/(2*epsilon)-pair.second->at(i)); maximum=std::max(maximum,error);
        if(error>2e-5)throw std::runtime_error("LoRA adapter gradient mismatch");
    }
    std::cout << "{\"gradient_check\":\"LoRA-adapter-gradients\",\"status\":\"passed\"}" << std::endl;
    return maximum;
}
void corrected_gemm_test(Arena &arena,Blas &blas) {
    size_t mark=arena.position;std::mt19937 rng(12345);std::uniform_real_distribution<float> random(-0.5f,0.5f);
    constexpr int m=128,n=192,k=256,batch=2;float maximum=0;
    for(bool frozen_a:{false,true})for(bool ta:{false,true})for(bool tb:{false,true}) {
        std::vector<float> ha(batch*m*k),hb(batch*k*n),hc(batch*m*n);
        for(float &x:ha){x=random(rng);if(frozen_a)x=std::round(x*128)/128;}
        for(float &x:hb){x=random(rng);if(!frozen_a)x=std::round(x*128)/128;}
        for(float &x:hc)x=random(rng);
        float *a=arena.alloc<float>(ha.size()),*b=arena.alloc<float>(hb.size());
        float *c=arena.alloc<float>(hc.size()),*reference=arena.alloc<float>(hc.size());
        cuda_check(cudaMemcpy(a,ha.data(),ha.size()*4,cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(b,hb.data(),hb.size()*4,cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(c,hc.data(),hc.size()*4,cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(reference,hc.data(),hc.size()*4,cudaMemcpyHostToDevice));
        blas.corrected_tf32=false;blas.compute=CUBLAS_COMPUTE_32F;
        blas.gemm(reference,a,b,m,n,k,ta,tb,0.7f,0.35f,batch);
        blas.corrected_tf32=true;blas.scratch_arena=&arena;blas.exact_tf32.insert(frozen_a?a:b);
        size_t scratch_mark=arena.position;
        blas.gemm(c,a,b,m,n,k,ta,tb,0.7f,0.35f,batch);
        if(arena.position!=scratch_mark)throw std::runtime_error("TF32 correction scratch not released");
        std::vector<float> actual(hc.size()),expected(hc.size());
        cuda_check(cudaMemcpy(actual.data(),c,hc.size()*4,cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(expected.data(),reference,hc.size()*4,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<hc.size();++i)maximum=std::max(maximum,std::abs(actual[i]-expected[i]));
        blas.exact_tf32.clear();arena.position=mark;
    }
    blas.corrected_tf32=false;blas.compute=CUBLAS_COMPUTE_32F;
    if(maximum>3e-5f)throw std::runtime_error("Corrected TF32 batched/transposed/accumulating GEMM mismatch");
    std::cout<<"{\"corrected_tf32_gemm_check\":\"passed\",\"max_abs\":"<<maximum<<"}"<<std::endl;
}
float bf16_round(float x) {
    uint32_t bits;std::memcpy(&bits,&x,4);
    bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000;
    std::memcpy(&x,&bits,4);return x;
}
void bf16_gemm_test(Arena &arena,Blas &blas) {
    size_t mark=arena.position;blas.scratch_arena=&arena;
    std::mt19937 rng(5678);std::uniform_real_distribution<float> random(-0.5f,0.5f);
    float maximum=0;
    for(auto shape:{std::vector<int>{128,192,256},std::vector<int>{13,17,7}})
    for(bool frozen:{false,true})for(bool ta:{false,true})for(bool tb:{false,true}) {
        int m=shape[0],n=shape[1],k=shape[2],batch=frozen?1:2;
        std::vector<float> ha(size_t(batch)*m*k),hb(size_t(batch)*k*n),hc(size_t(batch)*m*n);
        for(float &x:ha)x=random(rng);for(float &x:hb)x=random(rng);for(float &x:hc)x=random(rng);
        auto ar=ha,br=hb;for(float &x:ar)x=bf16_round(x);for(float &x:br)x=bf16_round(x);
        auto upload=[&](const std::vector<float> &h) {
            auto p=arena.alloc<float>(h.size());cuda_check(cudaMemcpy(p,h.data(),h.size()*4,cudaMemcpyHostToDevice));return p;
        };
        float *a=upload(ha),*b=upload(hb),*aq=upload(ar),*bq=upload(br),*c=upload(hc),*reference=upload(hc);
        blas.native_bf16=false;blas.compute=CUBLAS_COMPUTE_32F;
        blas.gemm(reference,aq,bq,m,n,k,ta,tb,0.7f,0.35f,batch);
        auto packed=arena.alloc<__nv_bfloat16>(hb.size());
        float_to_bf16<<<blocks(hb.size()),256>>>(b,packed,hb.size());
        size_t scratch_mark=arena.position;blas.native_bf16=true;
        if(frozen)blas.frozen_gemm(c,a,nullptr,packed,m,n,k,ta,tb,0.7f,0.35f);
        else blas.gemm(c,a,b,m,n,k,ta,tb,0.7f,0.35f,batch);
        if(arena.position!=scratch_mark)throw std::runtime_error("BF16 GEMM scratch not released");
        std::vector<float> actual(hc.size()),expected(hc.size());
        cuda_check(cudaMemcpy(actual.data(),c,hc.size()*4,cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(expected.data(),reference,hc.size()*4,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<hc.size();++i)maximum=std::max(maximum,std::abs(actual[i]-expected[i]));
        arena.position=mark;
    }
    blas.native_bf16=false;
    if(maximum>3e-5f)throw std::runtime_error("BF16 batched/transposed/accumulating GEMM mismatch");
    std::cout<<"{\"bf16_gemm_check\":\"passed\",\"max_abs\":"<<maximum<<"}"<<std::endl;
}
void bf16_lora_test(Arena &arena,Blas &blas) {
    size_t mark=arena.position;blas.native_bf16=true;blas.scratch_arena=&arena;
    constexpr int m=4,in=7,out=5,rank=3;
    std::mt19937 rng(998);std::uniform_real_distribution<float> random(-0.4f,0.4f);
    auto make=[&](int n) {std::vector<float> h(n);for(float &x:h)x=random(rng);return h;};
    auto upload=[&](const std::vector<float>&h) {
        auto p=arena.alloc<float>(h.size());cuda_check(cudaMemcpy(p,h.data(),h.size()*4,cudaMemcpyHostToDevice));return p;
    };
    auto hx=make(m*in),ha=make(rank*in),hb=make(out*rank),hw=make(out*in),hy=make(m*out);
    Linear l{upload(hw),upload(ha),upload(hb),arena.alloc<float>(ha.size()),arena.alloc<float>(hb.size()),in,out,rank,2};
    cuda_check(cudaMemset(l.ga,0,ha.size()*4));cuda_check(cudaMemset(l.gb,0,hb.size()*4));
    Graph g(arena,blas,true);auto x=g.view(upload(hx),m,in);auto y=g.linear(x,l);
    cuda_check(cudaMemcpy(g.grad(y),hy.data(),hy.size()*4,cudaMemcpyHostToDevice));g.backward();
    auto quant=[](std::vector<float> h){for(float &x:h)x=bf16_round(x);return h;};
    // Independent scalar CPU products, with explicit operand rounding at each
    // forward/backward GEMM boundary (the mixed-precision straight-through rule).
    auto product=[](const std::vector<float>&a,const std::vector<float>&b,int m,int n,int k,bool ta=false,bool tb=false,float scale=1) {
        std::vector<float> c(m*n);
        for(int i=0;i<m;++i)for(int j=0;j<n;++j) {
            double sum=0;for(int z=0;z<k;++z)sum+=double(a[ta?z*m+i:i*k+z])*b[tb?j*k+z:z*n+j];
            c[i*n+j]=float(scale*sum);
        }return c;
    };
    auto xp=quant(hx),ap=quant(ha),bp=quant(hb),wp=quant(hw),yp=quant(hy);
    auto sp=quant(product(xp,ap,m,rank,in,false,true));
    auto expected_y=product(xp,wp,m,out,in,false,true),added=product(sp,bp,m,out,rank,false,true,2);
    for(size_t i=0;i<added.size();++i)expected_y[i]+=added[i];
    auto gs=quant(product(yp,bp,m,rank,out,false,false,2));
    auto expected_gx=product(yp,wp,m,in,out),extra=product(gs,ap,m,in,rank);
    for(size_t i=0;i<extra.size();++i)expected_gx[i]+=extra[i];
    auto expected_ga=product(gs,xp,rank,in,m,true,false),expected_gb=product(yp,sp,out,rank,m,true,false,2);
    float maximum=0;
    for(auto pair:{std::make_pair(y->data,&expected_y),std::make_pair(x->grad,&expected_gx),
                   std::make_pair(l.ga,&expected_ga),std::make_pair(l.gb,&expected_gb)}) {
        std::vector<float> actual(pair.second->size());
        cuda_check(cudaMemcpy(actual.data(),pair.first,actual.size()*4,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<actual.size();++i)maximum=std::max(maximum,std::abs(actual[i]-pair.second->at(i)));
    }
    blas.native_bf16=false;arena.position=mark;
    if(maximum>1e-5)throw std::runtime_error("Cached BF16 LoRA CPU forward/backward mismatch");
    std::cout<<"{\"bf16_lora_cpu_check\":\"passed\",\"max_abs\":"<<maximum<<"}"<<std::endl;
}
int self_test() {
    Arena arena(128 * 1024 * 1024); Blas blas;
    const int t = 4, heads = 1, n = t * 128;
    std::mt19937 rng(17); std::uniform_real_distribution<float> random(-0.15f, 0.15f);
    std::vector<std::vector<float>> host(5);
    for (int i = 0; i < 5; ++i) {
        host[i].resize(i < 3 ? n : t);
        for (float &v : host[i]) v = i < 3 ? random(rng) : 0.65f + random(rng);
    }
    std::vector<float> objective(n); for (float &v : objective) v = random(rng);
    float *x[5], *gx[5];
    for (int i = 0; i < 5; ++i) {
        x[i] = arena.alloc<float>(host[i].size()); gx[i] = arena.alloc<float>(host[i].size());
        cuda_check(cudaMemcpy(x[i], host[i].data(), host[i].size() * 4, cudaMemcpyHostToDevice));
        cuda_check(cudaMemset(gx[i], 0, host[i].size() * 4));
    }
    float *out = arena.alloc<float>(n), *dy = arena.alloc<float>(n), *states = arena.alloc<float>(size_t(t) * 128 * 128);
    cuda_check(cudaMemcpy(dy, objective.data(), n * 4, cudaMemcpyHostToDevice));
    auto evaluate = [&] {
        delta_forward<<<heads, 128>>>(x[0], x[1], x[2], x[3], x[4], out, states, t, heads);
        std::vector<float> value(n); cuda_check(cudaMemcpy(value.data(), out, n * 4, cudaMemcpyDeviceToHost));
        return std::inner_product(value.begin(), value.end(), objective.begin(), 0.0);
    };
    evaluate();
    // Independent double-precision CPU recurrence also checks state orientation
    // and Qwen's 1/sqrt(128) query scaling in the forward pass.
    std::vector<double> cpu_state(128*128,0);
    std::vector<float> gpu_out(n); cuda_check(cudaMemcpy(gpu_out.data(),out,n*4,cudaMemcpyDeviceToHost));
    for(int row=0;row<t;++row) {
        for(double &s:cpu_state)s*=host[3][row];
        double error[128];
        for(int j=0;j<128;++j) {
            double prediction=0;
            for(int i=0;i<128;++i)prediction+=host[1][row*128+i]*cpu_state[i*128+j];
            error[j]=(host[2][row*128+j]-prediction)*host[4][row];
        }
        for(int i=0;i<128;++i)for(int j=0;j<128;++j)cpu_state[i*128+j]+=host[1][row*128+i]*error[j];
        for(int j=0;j<128;++j) {
            double expected=0;
            for(int i=0;i<128;++i)expected+=host[0][row*128+i]*cpu_state[i*128+j]/std::sqrt(128.0);
            if(std::abs(expected-gpu_out[row*128+j])>2e-7)throw std::runtime_error("DeltaNet CPU forward reference mismatch");
        }
    }
    delta_backward<<<heads, 128>>>(x[0], x[1], x[2], x[3], x[4], states, dy, gx[0], gx[1], gx[2], gx[3], gx[4], t, heads);
    double maximum = 0;
    for (int input = 0; input < 5; ++input) {
        std::vector<float> analytic(host[input].size());
        cuda_check(cudaMemcpy(analytic.data(), gx[input], analytic.size() * 4, cudaMemcpyDeviceToHost));
        for (int index : {0, int(analytic.size() / 2), int(analytic.size() - 1)}) {
            float original = host[input][index], epsilon = 0.002f, v = original + epsilon;
            cuda_check(cudaMemcpy(x[input] + index, &v, 4, cudaMemcpyHostToDevice)); double plus = evaluate();
            v = original - epsilon; cuda_check(cudaMemcpy(x[input] + index, &v, 4, cudaMemcpyHostToDevice)); double minus = evaluate();
            cuda_check(cudaMemcpy(x[input] + index, &original, 4, cudaMemcpyHostToDevice));
            double expected = (plus - minus) / (2 * epsilon), error = std::abs(expected - analytic[index]);
            maximum = std::max(maximum, error);
            if (error > 3e-5 + 0.003 * std::abs(expected)) throw std::runtime_error("DeltaNet finite-difference gradient mismatch");
        }
    }
    // Check the row-major cuBLAS interface against a known matrix product.
    float ha[6] = {1,2,3,4,5,6}, hb[6] = {7,8,9,10,11,12}, expected[4] = {58,64,139,154};
    float *a = arena.alloc<float>(6), *b = arena.alloc<float>(6), *c = arena.alloc<float>(4);
    cuda_check(cudaMemcpy(a, ha, 24, cudaMemcpyHostToDevice)); cuda_check(cudaMemcpy(b, hb, 24, cudaMemcpyHostToDevice));
    blas.gemm(c, a, b, 2, 2, 3); float hc[4]; cuda_check(cudaMemcpy(hc, c, 16, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 4; ++i) if (std::abs(hc[i] - expected[i]) > 1e-4) throw std::runtime_error("cuBLAS row-major mismatch");
    // Exact cross-entropy and its scaled gradient, including extreme logits.
    float logits_cpu[10]={1,2,3,4,5,1000,999,-1000,998,997}; uint32_t targets_cpu[2]={3,1};
    float *logits=arena.alloc<float>(10), *dlogits=arena.alloc<float>(10), *loss=arena.alloc<float>(2);
    uint32_t *target=arena.alloc<uint32_t>(2);
    cuda_check(cudaMemcpy(logits,logits_cpu,40,cudaMemcpyHostToDevice)); cuda_check(cudaMemcpy(target,targets_cpu,8,cudaMemcpyHostToDevice));
    cross_entropy<<<2,256>>>(logits,dlogits,target,loss,5,0.125f);
    float loss_cpu[2],grad_cpu[10];cuda_check(cudaMemcpy(loss_cpu,loss,8,cudaMemcpyDeviceToHost));cuda_check(cudaMemcpy(grad_cpu,dlogits,40,cudaMemcpyDeviceToHost));
    for(int row=0;row<2;++row) {
        double high=*std::max_element(logits_cpu+row*5,logits_cpu+(row+1)*5),sum=0;
        for(int i=0;i<5;++i)sum+=std::exp(logits_cpu[row*5+i]-high);
        double expected_loss=high+std::log(sum)-logits_cpu[row*5+targets_cpu[row]];
        if(std::abs(expected_loss-loss_cpu[row])>1e-4)throw std::runtime_error("Cross-entropy CPU reference mismatch");
        for(int i=0;i<5;++i) {
            double expected_grad=0.125*(std::exp(logits_cpu[row*5+i]-high)/sum-double(i==targets_cpu[row]));
            if(std::abs(expected_grad-grad_cpu[row*5+i])>1e-7)throw std::runtime_error("Cross-entropy scaled gradient mismatch");
        }
    }
    auto packed_gradient=arena.alloc<__nv_bfloat16>(10);
    cross_entropy_bf16<<<2,256>>>(logits,packed_gradient,target,loss,5,0.125f);
    bf16_to_float<<<1,32>>>(packed_gradient,dlogits,10);
    float converted[10],bf16_losses[2];
    cuda_check(cudaMemcpy(converted,dlogits,40,cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(bf16_losses,loss,8,cudaMemcpyDeviceToHost));
    for(int i=0;i<10;++i)if(converted[i]!=bf16_round(grad_cpu[i]))throw std::runtime_error("Fused BF16 cross-entropy gradient mismatch");
    for(int i=0;i<2;++i)if(bf16_losses[i]!=loss_cpu[i])throw std::runtime_error("BF16 gradient changed FP32 cross-entropy loss");
    std::cout<<"{\"bf16_cross_entropy_check\":\"passed\"}"<<std::endl;
    double graph_error = graph_tests(arena, blas);
    tiled_delta_test(arena);
    corrected_gemm_test(arena,blas);
    bf16_gemm_test(arena,blas);
    bf16_lora_test(arena,blas);
    std::cout << "{\"native_self_test\":\"passed\",\"delta_gradient_max_abs_error\":" << maximum
              << ",\"graph_gradient_max_abs_error\":" << graph_error << "}" << std::endl;
    return 0;
}
