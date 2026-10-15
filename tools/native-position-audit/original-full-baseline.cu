// Read-only baseline diagnostic. All model operations use the native codebase.
#define main archived_trainer_main
#include "../../native/qwen.cu"
#undef main

struct PrefixResult { std::vector<float> losses, hidden; uint64_t bytes=0; };

PrefixResult prefix_last(Arena &arena, Blas &blas, Parameters &p,
                        const std::vector<uint32_t> &tokens, int context,
                        const std::vector<uint32_t> &bytes, bool adapters, bool export_hidden=false, int seen=0) {
    if(!seen)seen=context;
    arena.position=p.persistent;
    int sequences=int(tokens.size()/(context+1)), rows=sequences*context;
    if(tokens.size()!=size_t(sequences)*(context+1) || !sequences)throw std::runtime_error("Invalid prefix batch");
    std::vector<uint32_t> inputs(rows),targets(sequences);
    PrefixResult result;result.losses.resize(sequences);
    for(int b=0;b<sequences;++b) {
        std::copy_n(tokens.begin()+size_t(b)*(context+1),context,inputs.begin()+size_t(b)*context);
        targets[b]=tokens[size_t(b)*(context+1)+seen];
        result.bytes+=bytes.at(targets[b]);
    }
    auto ids=arena.alloc<uint32_t>(rows),target=arena.alloc<uint32_t>(sequences);
    cuda_check(cudaMemcpy(ids,inputs.data(),inputs.size()*4,cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(target,targets.data(),targets.size()*4,cudaMemcpyHostToDevice));
    Graph g(arena,blas,false);g.context=context;g.batch=sequences;g.adapters=adapters;
    T hidden=forward(g,p,ids,rows);
    float *last=arena.alloc<float>(size_t(sequences)*Hidden);
    cuda_check(cudaMemcpy2DAsync(last,Hidden*4,hidden->data+size_t(seen-1)*Hidden,size_t(context)*Hidden*4,
        Hidden*4,sequences,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
    float *logits=arena.alloc<float>(size_t(sequences)*Vocab),*losses=arena.alloc<float>(sequences);
    p.head(blas,logits,last,sequences);
    cross_entropy<<<sequences,256>>>(logits,nullptr,target,losses,Vocab,0);
    cuda_check(cudaGetLastError());
    cuda_check(cudaStreamSynchronize(cudaStreamPerThread));
    cuda_check(cudaMemcpy(result.losses.data(),losses,sequences*4,cudaMemcpyDeviceToHost));
    for(float x:result.losses)if(!std::isfinite(x))throw std::runtime_error("Nonfinite baseline loss");
    if(export_hidden) {
        result.hidden.resize(size_t(sequences)*Hidden);
        cuda_check(cudaMemcpy(result.hidden.data(),last,result.hidden.size()*4,cudaMemcpyDeviceToHost));
    }
    return result;
}

int main() {
    try {
        cuda_check(cudaSetDevice(1));
        cudaDeviceProp device;cuda_check(cudaGetDeviceProperties(&device,1));
        if(std::string(device.name)!="NVIDIA RTX PRO 6000 Blackwell Workstation Edition")
            throw std::runtime_error("Diagnostic requires RTX PRO 6000; use CUDA_DEVICE_ORDER=PCI_BUS_ID");
        std::cout<<"{\"gpu\":\""<<device.name<<"\",\"device\":1,\"total_gib\":"<<double(device.totalGlobalMem)/GiB<<"}"<<std::endl;
        std::ifstream paths(".cache/qwen-hn1g/paths.txt");
        std::string model,manifest,train,eval,lengths;
        for(auto text:{&model,&manifest,&train,&eval,&lengths}) {
            std::getline(paths,*text);if(!text->empty() && text->back()=='\r')text->pop_back();
        }
        if(!paths)throw std::runtime_error("Missing model asset paths");
        Arena arena(20*GiB);Blas blas;blas.native_bf16=true;
        Parameters p(arena,model,manifest,16,32,3407,true);
        blas.scratch_arena=&arena;blas.workspace(arena,GiB);p.persistent=arena.position;
        std::vector<uint32_t> bytes(Vocab);
        std::ifstream byte_file(lengths,std::ios::binary);byte_file.read(reinterpret_cast<char *>(bytes.data()),bytes.size()*4);
        if(!byte_file)throw std::runtime_error("Missing byte table");
        const std::array<const char *,6> names={"v-4096","v-1024","v-256","v-seq-4096","v-seq-1024","v-seq-256"};
        for(size_t job=0;job<names.size();++job) {
            fs::path dir=fs::path("runs/qwen-hn1g/original-near256-fullcontext")/names[job];
            if(fs::exists(dir/"original.csv"))throw std::runtime_error("Baseline output already exists");
            std::ifstream file(dir/"inputs.bin",std::ios::binary);uint64_t magic;uint32_t samples,context;
            read_binary(file,magic);read_binary(file,samples);read_binary(file,context);
            int effective=job%3==0?16:job%3==1?64:256;
            int physical_size=job%3==0?4:job%3==1?8:32,seen=job%3==2?256:258;
            if(!file || magic!=0x514e494e50555431ull || samples!=64*effective || context!=(job%3==0?4096:job%3==1?1024:256))
                throw std::runtime_error("Invalid matched baseline inputs");
            std::ofstream output(dir/"original.csv");output<<std::setprecision(17)<<"step,processed_tokens,input_tokens_seen,last_nll,target_bytes,bpb\n";
            auto begun=std::chrono::steady_clock::now();
            for(int update=0;update<64;++update) {
                double total_nll=0;uint64_t total_bytes=0;
                for(int lane=0;lane<effective;lane+=physical_size) {
                    int physical=std::min(physical_size,effective-lane);
                    std::vector<uint32_t> tokens(size_t(physical)*(context+1));
                    file.read(reinterpret_cast<char *>(tokens.data()),tokens.size()*4);
                    if(!file)throw std::runtime_error("Truncated baseline inputs");
                    for(auto id:tokens)if(id>=Vocab)throw std::runtime_error("Invalid baseline token");
                    bool audit=update==0 && lane==0;
                    auto result=prefix_last(arena,blas,p,tokens,context,bytes,false,audit,seen);
                    if(audit) {
                        auto zero=prefix_last(arena,blas,p,tokens,context,bytes,true,true,seen);
                        if(result.losses!=zero.losses || result.hidden!=zero.hidden || result.bytes!=zero.bytes)
                            throw std::runtime_error("Zero LoRA changes original model");
                        // Compare the last-only output head against the trainer's full loss path.
                        Runner runner(arena,blas,p,lengths,context,1);
                        Metrics full=runner.batch(tokens,false);
                        double expected=std::accumulate(result.losses.begin(),result.losses.end(),0.);
                        double error=std::abs((context==256?full.last_nll:full.position_nll[5])-expected)/physical;
                        if(error>1e-4 || (context==256 && full.last_bytes!=result.bytes))throw std::runtime_error("Baseline head disagrees with trainer loss");
                        std::cout<<"{\"run\":\""<<names[job]<<"\",\"zero_lora_exact\":true,\"trainer_loss_mean_abs_error\":"<<error<<"}"<<std::endl;
                    }
                    total_nll+=std::accumulate(result.losses.begin(),result.losses.end(),0.);total_bytes+=result.bytes;
                }
                int step=1177+update;
                output<<step<<','<<int64_t(step)*65536<<','<<seen<<','<<total_nll<<','<<total_bytes<<','<<total_nll/std::log(2.)/total_bytes<<'\n';
                output.flush();
                if((update+1)%8==0)std::cout<<"{\"run\":\""<<names[job]<<"\",\"updates_evaluated\":"<<update+1
                    <<",\"elapsed_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-begun).count()<<"}"<<std::endl;
            }
            if(file.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Trailing baseline input bytes");
        }
        std::cout<<"{\"complete\":true,\"peak_arena_gib\":"<<double(arena.peak)/GiB<<"}"<<std::endl;
        return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 1;}
}
