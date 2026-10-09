#pragma once
// Exact recurrent DeltaNet, tiled across independent value columns. State is
// checkpointed every four tokens and recomputed in registers during backward.
// A warp owns four value columns; its lanes distribute the 128 key coordinates.
constexpr int DeltaChunk = 4, DeltaValues = 4, DeltaGroups = 8;
template<int N> __device__ inline void warp_sum_vector(float (&values)[N]) {
    #pragma unroll
    for(int distance=16;distance;distance/=2) {
        #pragma unroll
        for(int j=0;j<N;++j)values[j]+=__shfl_xor_sync(0xffffffff,values[j],distance);
    }
}
__device__ inline size_t delta_state_at(int seq, int row, int head, int value, int key, int rows, int heads) {
    return ((((size_t(seq)*rows+row)*heads+head)*128+value)*128+key);
}
__global__ void delta_tiled_forward(const float *q,const float *k,const float *v,const float *d,
    const float *beta,float *y,float *checkpoint,int t,int heads) {
    int seq=blockIdx.x/(heads*8), head=blockIdx.x/8%heads;
    int group=blockIdx.x%8*4+threadIdx.x/32, first=group*4, lane=threadIdx.x%32;
    int chunks=(t+DeltaChunk-1)/DeltaChunk;
    float state[4][4]={};
    for(int row=0;row<t;++row) {
        if(checkpoint && row%DeltaChunk==0) {
            #pragma unroll
            for(int j=0;j<4;++j) {
                #pragma unroll
                for(int a=0;a<4;++a)
                    checkpoint[delta_state_at(seq,row/DeltaChunk,head,first+j,lane+32*a,chunks,heads)]=state[j][a];
            }
        }
        int base=((seq*t+row)*heads+head)*128;
        float decay=d[(seq*t+row)*heads+head], b=beta[(seq*t+row)*heads+head];
        float keys[4],queries[4];
        #pragma unroll
        for(int a=0;a<4;++a) { keys[a]=k[base+lane+32*a];queries[a]=q[base+lane+32*a]; }
        float dots[4]={},outputs[4]={};
        #pragma unroll
        for(int j=0;j<4;++j) {
            #pragma unroll
            for(int a=0;a<4;++a) {state[j][a]*=decay;dots[j]+=keys[a]*state[j][a];}
        }
        warp_sum_vector(dots);
        #pragma unroll
        for(int j=0;j<4;++j) {
            float error=(v[base+first+j]-dots[j])*b;
            #pragma unroll
            for(int a=0;a<4;++a) {state[j][a]+=keys[a]*error;outputs[j]+=queries[a]*state[j][a];}
        }
        warp_sum_vector(outputs);
        if(lane==0) {
            #pragma unroll
            for(int j=0;j<4;++j)y[base+first+j]=outputs[j]*0.08838834764831845f;
        }
    }
}
__global__ void delta_tiled_backward(const float *q,const float *k,const float *v,const float *d,
    const float *beta,const float *checkpoint,const float *dy,float *dv,
    float *history,float *errors,float2 *qk_parts,float2 *db_parts,int t,int heads) {
    int seq=blockIdx.x/(heads*8), head=blockIdx.x/8%heads;
    int group=blockIdx.x%8*4+threadIdx.x/32, first=group*4,lane=threadIdx.x%32;
    int chunks=(t+DeltaChunk-1)/DeltaChunk;
    __shared__ float2 partial_qk[4][128],partial_db[4];
    int warp=threadIdx.x/32,tile=blockIdx.x%8;
    float adj[4][4]={};
    for(int chunk=chunks-1;chunk>=0;--chunk) {
        float state[4][4];
        #pragma unroll
        for(int j=0;j<4;++j) {
            #pragma unroll
            for(int a=0;a<4;++a)
                state[j][a]=checkpoint[delta_state_at(seq,chunk,head,first+j,lane+32*a,chunks,heads)];
        }
        int begin=chunk*DeltaChunk;
        float row_states[DeltaChunk][4][4],row_errors[DeltaChunk][4];
        // Each warp replays its own state tile; block barriers combine its
        // derivative with the other three warps before writing to VRAM.
        #pragma unroll
        for(int r=0;r<DeltaChunk;++r) {
            int row=begin+r;if(row>=t)continue;
            int base=((seq*t+row)*heads+head)*128;
            float decay=d[(seq*t+row)*heads+head],b=beta[(seq*t+row)*heads+head];
            float keys[4];
            #pragma unroll
            for(int a=0;a<4;++a)keys[a]=k[base+lane+32*a];
            float dots[4]={};
            #pragma unroll
            for(int j=0;j<4;++j) {
                #pragma unroll
                for(int a=0;a<4;++a){state[j][a]*=decay;dots[j]+=keys[a]*state[j][a];}
            }
            warp_sum_vector(dots);
            #pragma unroll
            for(int j=0;j<4;++j) {
                float error=v[base+first+j]-dots[j];
                row_errors[r][j]=error;
                #pragma unroll
                for(int a=0;a<4;++a) {
                    state[j][a]+=keys[a]*b*error;
                    row_states[r][j][a]=state[j][a];
                }
            }
        }
        #pragma unroll
        for(int r=DeltaChunk-1;r>=0;--r) {
            int row=begin+r;if(row>=t)continue;
            int base=((seq*t+row)*heads+head)*128;
            float decay=d[(seq*t+row)*heads+head],b=beta[(seq*t+row)*heads+head];
            float keys[4],queries[4],gq[4]={},gk[4]={},decay_grad=0,beta_grad=0;
            #pragma unroll
            for(int a=0;a<4;++a){keys[a]=k[base+lane+32*a];queries[a]=q[base+lane+32*a];}
            float deltas[4]={},output_grads[4];
            #pragma unroll
            for(int j=0;j<4;++j) {
                output_grads[j]=dy[base+first+j]*0.08838834764831845f;
                #pragma unroll
                for(int a=0;a<4;++a){adj[j][a]+=queries[a]*output_grads[j];deltas[j]+=keys[a]*adj[j][a];}
            }
            warp_sum_vector(deltas);
            #pragma unroll
            for(int j=0;j<4;++j) {
                float delta_grad=deltas[j],output_grad=output_grads[j];
                float error=row_errors[r][j];
                float value_grad=b*delta_grad;
                if(lane==0)dv[base+first+j]+=value_grad;
                beta_grad+=delta_grad*error;
                #pragma unroll
                for(int a=0;a<4;++a) {
                    int key=lane+32*a;
                    float current=row_states[r][j][a];
                    float prev=r==0?checkpoint[delta_state_at(seq,chunk,head,first+j,key,chunks,heads)]:row_states[r-1][j][a];
                    gq[a]+=output_grad*current;
                    gk[a]+=adj[j][a]*b*error-prev*decay*value_grad;
                    float memory_grad=adj[j][a]-keys[a]*value_grad;
                    decay_grad+=memory_grad*prev;adj[j][a]=decay*memory_grad;
                }
            }
            #pragma unroll
            for(int a=0;a<4;++a) {
                partial_qk[warp][lane+32*a]=make_float2(gq[a],gk[a]);
            }
            decay_grad=warp_sum(decay_grad);
            if(lane==0)partial_db[warp]=make_float2(decay_grad,beta_grad);
            __syncthreads();
            float2 total=make_float2(0,0);
            #pragma unroll
            for(int w=0;w<4;++w){float2 p=partial_qk[w][threadIdx.x];total.x+=p.x;total.y+=p.y;}
            qk_parts[((((size_t(seq)*heads+head)*DeltaGroups+tile)*t+row)*128+threadIdx.x)]=total;
            if(threadIdx.x==0) {
                total=make_float2(0,0);
                #pragma unroll
                for(int w=0;w<4;++w){total.x+=partial_db[w].x;total.y+=partial_db[w].y;}
                db_parts[(((size_t(seq)*heads+head)*DeltaGroups+tile)*t+row)]=total;
            }
            __syncthreads();
        }
    }
}
__global__ void delta_reduce(const float2 *qk,const float2 *db,float *dq,float *dk,float *dd,float *d_beta,
    int t,int heads,int batch) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,n=batch*t*heads*128;
    if(i>=n)return;
    int key=i%128,head=i/128%heads,row=i/(128*heads)%t,seq=i/(128*heads*t);
    size_t start=(((size_t(seq)*heads+head)*DeltaGroups)*t+row)*128+key;
    float a=0,b=0;
    #pragma unroll
    for(int group=0;group<DeltaGroups;++group){float2 p=qk[start+size_t(group)*t*128];a+=p.x;b+=p.y;}
    dq[i]+=a;dk[i]+=b;
    if(key==0) {
        a=b=0;start=((size_t(seq)*heads+head)*DeltaGroups)*t+row;
        #pragma unroll
        for(int group=0;group<DeltaGroups;++group){float2 p=db[start+size_t(group)*t];a+=p.x;b+=p.y;}
        dd[(seq*t+row)*heads+head]+=a;d_beta[(seq*t+row)*heads+head]+=b;
    }
}
