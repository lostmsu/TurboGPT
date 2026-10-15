#define main archived_trainer_main
#include "../../native/qwen.cu"
#undef main

std::vector<float> position_losses(Arena &arena,Blas &blas,Parameters &p,
    const std::vector<uint32_t> &tokens,int context,int seen) {
    arena.position=p.persistent;
    int batch=int(tokens.size()/(context+1)),rows=batch*context;
    std::vector<uint32_t> input(rows),target(batch);
    for(int b=0;b<batch;++b) {
        std::copy_n(tokens.begin()+size_t(b)*(context+1),context,input.begin()+size_t(b)*context);
        target[b]=tokens[size_t(b)*(context+1)+seen];
    }
    auto ids=arena.alloc<uint32_t>(rows),targets=arena.alloc<uint32_t>(batch);
    cuda_check(cudaMemcpy(ids,input.data(),rows*4,cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(targets,target.data(),batch*4,cudaMemcpyHostToDevice));
    Graph graph(arena,blas,false);graph.context=context;graph.batch=batch;graph.adapters=false;
    T hidden=forward(graph,p,ids,rows);
    auto selected=arena.alloc<float>(size_t(batch)*Hidden);
    cuda_check(cudaMemcpy2DAsync(selected,Hidden*4,hidden->data+size_t(seen-1)*Hidden,size_t(context)*Hidden*4,
        Hidden*4,batch,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
    auto logits=arena.alloc<float>(size_t(batch)*Vocab),losses=arena.alloc<float>(batch);
    p.head(blas,logits,selected,batch);
    cross_entropy<<<batch,256>>>(logits,nullptr,targets,losses,Vocab,0);
    cuda_check(cudaStreamSynchronize(cudaStreamPerThread));
    std::vector<float> result(batch);cuda_check(cudaMemcpy(result.data(),losses,batch*4,cudaMemcpyDeviceToHost));
    return result;
}
int main() {
    try {
        cuda_check(cudaSetDevice(1));
        cudaDeviceProp device;cuda_check(cudaGetDeviceProperties(&device,1));
        if(std::string(device.name)!="NVIDIA RTX PRO 6000 Blackwell Workstation Edition")
            throw std::runtime_error("Prefix diagnostic requires RTX PRO 6000");
        std::cout<<"{\"gpu\":\""<<device.name<<"\",\"device\":1}"<<std::endl;
        std::ifstream paths(".cache/qwen-hn1g/paths.txt");std::string model,manifest,train,eval,lengths;
        for(auto text:{&model,&manifest,&train,&eval,&lengths}) {
            std::getline(paths,*text);if(!text->empty() && text->back()=='\r')text->pop_back();
        }
        Arena arena(12*GiB);Blas blas;blas.native_bf16=true;
        Parameters p(arena,model,manifest,16,32,3407,true);blas.scratch_arena=&arena;
        blas.workspace(arena,GiB);p.persistent=arena.position;Tokens dataset(train);
        const std::array<const char *,4> names={"v-4096","v-1024","v-seq-4096","v-seq-1024"};
        for(int job=0;job<4;++job) {
            int context=job%2?1024:4096,seen=258;
            fs::path dir=fs::path("runs/qwen-hn1g/original-near256")/names[job];
            std::ifstream positions(dir/"starts.u64",std::ios::binary);
            std::vector<uint32_t> full,prefix;
            for(int b=0;b<2;++b) {
                uint64_t start;read_binary(positions,start);if(!positions)throw std::runtime_error("Missing replay starts");
                auto window=dataset.block(start,context);
                full.insert(full.end(),window.begin(),window.end());
                prefix.insert(prefix.end(),window.begin(),window.begin()+seen+1);
            }
            auto a=position_losses(arena,blas,p,full,context,seen),b=position_losses(arena,blas,p,prefix,seen,seen);
            double maximum=0;for(size_t i=0;i<a.size();++i)maximum=std::max(maximum,std::abs(double(a[i])-b[i]));
            std::cout<<std::setprecision(17)<<"{\"run\":\""<<names[job]<<"\",\"samples\":2,\"prefix_vs_full_max_loss_error_nats\":"<<maximum<<"}"<<std::endl;
            if(!std::isfinite(maximum))throw std::runtime_error("Nonfinite prefix diagnostic");
        }
        std::cout<<"{\"prefix_diagnostic\":\"complete\"}"<<std::endl;return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 1;}
}
