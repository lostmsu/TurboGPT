#pragma once
#include "optim.h"
// Optimizer equations and bit-preserving weight representation reused from
// ../turboGPT/cuda/step.cuh (precise_weight/write_weight, adamw, Muon phases).
// The original fixed Width/64 WMMA layout is replaced by shape-grouped cuBLAS
// products because LoRA matrices have a short dimension of 16, not Width.
namespace original_optimizer {
struct DeviceState { __nv_bfloat16 *weight_high; uint16_t *weight_low; };
__device__ __forceinline__ float precise_weight(const DeviceState &d, int i) {
    return __uint_as_float((unsigned(__bfloat16_as_ushort(d.weight_high[i])) << 16) |
                           d.weight_low[i]);
}
__device__ __forceinline__ void write_weight(const DeviceState &d, int i, float value) {
    unsigned bits = __float_as_uint(value);
    d.weight_high[i] = __ushort_as_bfloat16(bits >> 16);
    d.weight_low[i] = uint16_t(bits);
}
__device__ __forceinline__ float round_bf16(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}
__global__ void pack_weights(DeviceState d,const float *values,size_t n) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)
        write_weight(d,int(i),values[i]);
}
__global__ void mantissa_adamw(DeviceState d,float *values,const float *gradient,float *momentum,float *variance,
    size_t n,float lr,float clip,float beta1,float beta2,float bc1,float bc2,float weight_decay,float epsilon) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
        float g=gradient[i]*clip;
        float m=momentum[i]+(1-beta1)*(g-momentum[i]);
        float v=beta2*variance[i]+(1-beta2)*g*g;
        momentum[i]=m;variance[i]=v;
        float w=precise_weight(d,int(i))*(1-lr*weight_decay);
        float next=w-(lr/bc1)*m/(sqrtf(v)/sqrtf(bc2)+epsilon);
        write_weight(d,int(i),next);values[i]=next;
    }
}
struct Matrix { int weight,rows,cols;size_t x; };
struct Chunk { int matrix,begin,end; };
__device__ __forceinline__ size_t muon_index(const Matrix &m,int e) {
    return m.rows>m.cols?size_t(e%m.cols)*m.rows+e/m.cols:size_t(e);
}
__global__ void muon_prepare(const Matrix *matrices,const Chunk *chunks,const float *gradient,float *momentum,
    __nv_bfloat16 *x,float *norms,float clip,float momentum_rate) {
    Chunk chunk=chunks[blockIdx.x];Matrix m=matrices[chunk.matrix];float squares=0;
    for(int e=chunk.begin+threadIdx.x;e<chunk.end;e+=blockDim.x) {
        int i=m.weight+e;float g=gradient[i]*clip;
        float mo=momentum[i]+(1-momentum_rate)*(g-momentum[i]);momentum[i]=mo;
        __nv_bfloat16 u=__float2bfloat16_rn(g+momentum_rate*(mo-g));
        x[m.x+muon_index(m,e)]=u;squares+=float(u)*float(u);
    }
    squares=block_sum(squares);
    if(threadIdx.x==0)atomicAdd(norms+chunk.matrix,squares);
}
__global__ void muon_normalize(const Matrix *matrices,const Chunk *chunks,__nv_bfloat16 *x,const float *norms) {
    Chunk chunk=chunks[blockIdx.x];Matrix m=matrices[chunk.matrix];
    float norm=fmaxf(round_bf16(sqrtf(norms[chunk.matrix])),round_bf16(1e-7f));
    for(int e=chunk.begin+threadIdx.x;e<chunk.end;e+=blockDim.x)
        x[m.x+e]=__float2bfloat16_rn(float(x[m.x+e])/norm);
}
__global__ void muon_polynomial(const __nv_bfloat16 *a,const float *square,__nv_bfloat16 *b,size_t n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)b[i]=__float2bfloat16_rn(-4.775f*float(a[i])+2.0315f*square[i]);
}
__global__ void muon_combine(const __nv_bfloat16 *x,__nv_bfloat16 *next,size_t n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)next[i]=__float2bfloat16_rn(3.4445f*float(x[i])+float(next[i]));
}
__global__ void muon_apply(DeviceState d,const Matrix *matrices,const Chunk *chunks,const __nv_bfloat16 *x,
    float *values,float lr,float weight_decay) {
    Chunk chunk=chunks[blockIdx.x];Matrix m=matrices[chunk.matrix];
    float adjusted=lr*sqrtf(fmaxf(1.f,float(m.rows)/m.cols));
    for(int e=chunk.begin+threadIdx.x;e<chunk.end;e+=blockDim.x) {
        int i=m.weight+e;float w=precise_weight(d,i)*(1-lr*weight_decay);
        float next=w-adjusted*float(x[m.x+muon_index(m,e)]);
        write_weight(d,i,next);values[i]=next;
    }
}
} // namespace original_optimizer

struct Optimizer {
    Arena &arena;Blas &blas;Parameters &p;std::string kind;
    original_optimizer::DeviceState bits;
    original_optimizer::Matrix *matrices=nullptr;
    original_optimizer::Chunk *chunks=nullptr;
    __nv_bfloat16 *x[2]={},*gram=nullptr,*poly=nullptr;
    float *norms=nullptr,*square=nullptr;
    struct Group {int short_dim,long_dim,count;size_t x,gram;};
    std::vector<Group> groups;
    int matrix_count=0,chunk_count=0;size_t gram_count=0;
    Optimizer(Arena &a,Blas &b,Parameters &parameters,const std::string &name):arena(a),blas(b),p(parameters),kind(name) {
        if(kind!="muon" && kind!="mantissa-adamw" && kind!="adamw")throw std::runtime_error("Unknown optimizer");
        if(kind=="adamw")return;
        bits.weight_high=arena.alloc<__nv_bfloat16>(p.count);bits.weight_low=arena.alloc<uint16_t>(p.count);
        if(kind=="muon") {
            std::map<std::pair<int,int>,std::vector<original_optimizer::Matrix>> by_shape;
            for(auto &[name,l]:p.linears)for(auto m:{original_optimizer::Matrix{int(l.a-p.values),l.rank,l.input,0},
                original_optimizer::Matrix{int(l.b-p.values),l.output,l.rank,0}})
                by_shape[{std::min(m.rows,m.cols),std::max(m.rows,m.cols)}].push_back(m);
            std::vector<original_optimizer::Matrix> host;std::vector<original_optimizer::Chunk> work;
            size_t cursor=0;
            for(auto &[shape,items]:by_shape) {
                Group g{shape.first,shape.second,int(items.size()),cursor,gram_count};groups.push_back(g);
                gram_count+=size_t(g.count)*g.short_dim*g.short_dim;
                for(auto m:items) {
                    m.x=cursor;cursor+=size_t(m.rows)*m.cols;
                    int id=int(host.size());host.push_back(m);
                    for(int begin=0;begin<m.rows*m.cols;begin+=4096)
                        work.push_back({id,begin,std::min(begin+4096,m.rows*m.cols)});
                }
            }
            if(cursor!=p.count)throw std::runtime_error("Muon parameter layout mismatch");
            matrix_count=int(host.size());chunk_count=int(work.size());
            matrices=arena.alloc<original_optimizer::Matrix>(matrix_count);chunks=arena.alloc<original_optimizer::Chunk>(chunk_count);
            cuda_check(cudaMemcpy(matrices,host.data(),host.size()*sizeof(host[0]),cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(chunks,work.data(),work.size()*sizeof(work[0]),cudaMemcpyHostToDevice));
            x[0]=arena.alloc<__nv_bfloat16>(p.count);x[1]=arena.alloc<__nv_bfloat16>(p.count);
            gram=arena.alloc<__nv_bfloat16>(gram_count);poly=arena.alloc<__nv_bfloat16>(gram_count);
            square=arena.alloc<float>(gram_count);norms=arena.alloc<float>(matrix_count);
        }
        p.persistent=arena.position;sync_bits();
    }
    void sync_bits() {
        if(kind!="adamw")original_optimizer::pack_weights<<<1024,256>>>(bits,p.values,p.count);
    }
    template<class Output> void bf16_gemm(Output *c,const __nv_bfloat16 *a,const __nv_bfloat16 *b,int m,int n,int k,
        bool ta,bool tb,int batch,float beta=0) {
        float one=1;
        blas_check(cublasGemmStridedBatchedEx(blas.handle,tb?CUBLAS_OP_T:CUBLAS_OP_N,ta?CUBLAS_OP_T:CUBLAS_OP_N,
            n,m,k,&one,b,CUDA_R_16BF,tb?k:n,int64_t(k)*n,a,CUDA_R_16BF,ta?m:k,int64_t(m)*k,
            &beta,c,sizeof(Output)==4?CUDA_R_32F:CUDA_R_16BF,n,int64_t(m)*n,batch,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    }
    void step(int step,float lr,float clip,float momentum_rate=0.95f) {
        using namespace original_optimizer;
        if(kind=="adamw") {
            adamw<<<1024,256>>>(p.values,p.gradients,p.moment1,p.moment2,p.count,lr,clip,
                1.f-std::pow(0.9f,step),1.f-std::pow(0.999f,step));return;
        }
        if(kind=="mantissa-adamw") {
            mantissa_adamw<<<1024,256>>>(bits,p.values,p.gradients,p.moment1,p.moment2,p.count,lr,clip,0.9f,0.999f,
                1.f-std::pow(0.9f,step),1.f-std::pow(0.999f,step),0.01f,1e-8f);return;
        }
        cuda_check(cudaMemset(norms,0,matrix_count*4));
        muon_prepare<<<chunk_count,256>>>(matrices,chunks,p.gradients,p.moment1,x[0],norms,clip,momentum_rate);
        muon_normalize<<<chunk_count,256>>>(matrices,chunks,x[0],norms);
        for(int iteration=0;iteration<5;++iteration) {
            for(auto &g:groups)bf16_gemm(gram+g.gram,x[iteration%2]+g.x,x[iteration%2]+g.x,
                g.short_dim,g.short_dim,g.long_dim,false,true,g.count);
            for(auto &g:groups)bf16_gemm(square+g.gram,gram+g.gram,gram+g.gram,
                g.short_dim,g.short_dim,g.short_dim,false,false,g.count);
            muon_polynomial<<<blocks(gram_count),256>>>(gram,square,poly,gram_count);
            cuda_check(cudaMemcpyAsync(x[(iteration+1)%2],x[iteration%2],p.count*2,cudaMemcpyDeviceToDevice));
            for(auto &g:groups)bf16_gemm(x[(iteration+1)%2]+g.x,poly+g.gram,x[iteration%2]+g.x,
                g.short_dim,g.long_dim,g.short_dim,false,false,g.count,3.4445f);
        }
        muon_apply<<<chunk_count,256>>>(bits,matrices,chunks,x[1],p.values,lr,0.01f);
    }
};
