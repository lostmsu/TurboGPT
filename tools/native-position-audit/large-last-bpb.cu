// Paired Monte Carlo evaluation, no optimizer or updates. Fixed inline settings.
#define main archived_trainer_main
#include "../../native/qwen.cu"
#undef main

constexpr int EvaluationContext=4096,EvaluationSamples=8192,EvaluationBatch=16;
constexpr uint32_t EvaluationSeed=20261010;
const fs::path EvaluationOutput="runs/qwen-hn1g/last4096-shared-8192";

std::vector<float> last_losses(Arena &arena,Blas &blas,Parameters &p,
    const std::vector<uint32_t> &tokens,bool adapters) {
    arena.position=p.persistent;
    int batch=int(tokens.size()/(EvaluationContext+1)),rows=batch*EvaluationContext;
    std::vector<uint32_t> inputs(rows),targets(batch);
    for(int b=0;b<batch;++b) {
        std::copy_n(tokens.begin()+size_t(b)*(EvaluationContext+1),EvaluationContext,inputs.begin()+size_t(b)*EvaluationContext);
        targets[b]=tokens[size_t(b)*(EvaluationContext+1)+EvaluationContext];
    }
    auto ids=arena.alloc<uint32_t>(rows),target=arena.alloc<uint32_t>(batch);
    cuda_check(cudaMemcpy(ids,inputs.data(),rows*4,cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(target,targets.data(),batch*4,cudaMemcpyHostToDevice));
    Graph graph(arena,blas,false);graph.context=EvaluationContext;graph.batch=batch;graph.adapters=adapters;
    T hidden=forward(graph,p,ids,rows);
    auto last=arena.alloc<float>(size_t(batch)*Hidden);
    cuda_check(cudaMemcpy2DAsync(last,Hidden*4,hidden->data+size_t(EvaluationContext-1)*Hidden,size_t(EvaluationContext)*Hidden*4,
        Hidden*4,batch,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
    auto logits=arena.alloc<float>(size_t(batch)*Vocab),losses=arena.alloc<float>(batch);
    p.head(blas,logits,last,batch);
    cross_entropy<<<batch,256>>>(logits,nullptr,target,losses,Vocab,0);
    cuda_check(cudaGetLastError());cuda_check(cudaStreamSynchronize(cudaStreamPerThread));
    std::vector<float> host(batch);cuda_check(cudaMemcpy(host.data(),losses,batch*4,cudaMemcpyDeviceToHost));
    for(float x:host)if(!std::isfinite(x))throw std::runtime_error("Nonfinite evaluation NLL");
    return host;
}

int main() {
    try {
        cuda_check(cudaSetDevice(1));cudaDeviceProp device;cuda_check(cudaGetDeviceProperties(&device,1));
        if(std::string(device.name)!="NVIDIA RTX PRO 6000 Blackwell Workstation Edition")
            throw std::runtime_error("Requires RTX PRO 6000 and CUDA_DEVICE_ORDER=PCI_BUS_ID");
        std::cout<<"{\"gpu\":\""<<device.name<<"\",\"device\":1,\"context\":"<<EvaluationContext
            <<",\"samples\":"<<EvaluationSamples<<",\"batch\":"<<EvaluationBatch<<",\"seed\":"<<EvaluationSeed<<"}"<<std::endl;
        std::ifstream paths(".cache/qwen-hn1g/paths.txt");std::string model,manifest,train,eval,lengths;
        for(auto text:{&model,&manifest,&train,&eval,&lengths}) {
            std::getline(paths,*text);if(!text->empty() && text->back()=='\r')text->pop_back();
        }
        if(!paths)throw std::runtime_error("Missing model assets");
        fs::create_directories(EvaluationOutput);
        for(auto name:{"base","regular","sequential"})if(fs::exists(EvaluationOutput/(std::string(name)+".f32")))
            throw std::runtime_error("Evaluation scores already exist");
        Tokens corpus(train);std::mt19937 sampling(EvaluationSeed);
        std::uniform_int_distribution<uint64_t> distribution(0,corpus.count-EvaluationContext-1);
        std::vector<uint32_t> windows(size_t(EvaluationSamples)*(EvaluationContext+1)),target_bytes(EvaluationSamples);
        std::vector<uint64_t> starts(EvaluationSamples);
        std::vector<uint32_t> bytes(Vocab);std::ifstream byte_file(lengths,std::ios::binary);
        byte_file.read(reinterpret_cast<char *>(bytes.data()),bytes.size()*4);if(!byte_file)throw std::runtime_error("Missing byte lengths");
        for(int i=0;i<EvaluationSamples;++i) {
            starts[i]=distribution(sampling);auto window=corpus.block(starts[i],EvaluationContext);
            std::copy(window.begin(),window.end(),windows.begin()+size_t(i)*(EvaluationContext+1));
            target_bytes[i]=bytes.at(window.back());if(!target_bytes[i])throw std::runtime_error("Zero-length target");
        }
        {std::ofstream out(EvaluationOutput/"starts.u64",std::ios::binary);out.write(reinterpret_cast<char *>(starts.data()),starts.size()*8);}
        {std::ofstream out(EvaluationOutput/"target_bytes.u32",std::ios::binary);out.write(reinterpret_cast<char *>(target_bytes.data()),target_bytes.size()*4);}
        {std::ofstream out(EvaluationOutput/"inputs.bin",std::ios::binary);uint64_t magic=0x514e494e50555431ull;
            uint32_t samples=EvaluationSamples,context=EvaluationContext;write_binary(out,magic);write_binary(out,samples);write_binary(out,context);
            out.write(reinterpret_cast<char *>(windows.data()),windows.size()*4);}
        Arena arena(32*GiB);Blas blas;blas.native_bf16=true;
        Parameters p(arena,model,manifest,16,32,3407,true);blas.scratch_arena=&arena;
        blas.workspace(arena,GiB);p.persistent=arena.position;
        const std::array<const char *,3> names={"base","regular","sequential"};
        const std::array<const char *,3> checkpoints={"","runs/qwen-hn1g/v-4096/checkpoint.bin","runs/qwen-hn1g/v-seq-4096/checkpoint.bin"};
        uint64_t total_bytes=std::accumulate(target_bytes.begin(),target_bytes.end(),uint64_t(0));
        for(int model_index=0;model_index<3;++model_index) {
            if(model_index) {
                std::mt19937 unused_rng;int step=restore(checkpoints[model_index],p,EvaluationContext,1,unused_rng,EvaluationBatch,"muon");
                if(step!=1240)throw std::runtime_error("Expected the final checkpoint at update 1240");
                std::cout<<"{\"model\":\""<<names[model_index]<<"\",\"restored_step\":"<<step<<"}"<<std::endl;
            }
            std::ofstream scores(EvaluationOutput/(std::string(names[model_index])+".f32"),std::ios::binary);
            auto begun=std::chrono::steady_clock::now();double total_nll=0;
            for(int index=0;index<EvaluationSamples;index+=EvaluationBatch) {
                auto first=windows.begin()+size_t(index)*(EvaluationContext+1);
                std::vector<uint32_t> batch(first,first+size_t(EvaluationBatch)*(EvaluationContext+1));
                auto losses=last_losses(arena,blas,p,batch,model_index!=0);
                if(index==0) {
                    Runner runner(arena,blas,p,lengths,EvaluationContext,1);
                    Metrics full=runner.batch(batch,false);
                    double observed=std::accumulate(losses.begin(),losses.end(),0.);
                    double error=std::abs(full.last_nll-observed)/EvaluationBatch;
                    if(error>1e-4)throw std::runtime_error("Last-only head disagrees with full trainer loss");
                    std::cout<<"{\"model\":\""<<names[model_index]<<"\",\"trainer_loss_mean_abs_error\":"<<error<<"}"<<std::endl;
                }
                scores.write(reinterpret_cast<char *>(losses.data()),losses.size()*4);
                if(!scores)throw std::runtime_error("Failed writing evaluation scores");
                total_nll+=std::accumulate(losses.begin(),losses.end(),0.);
                if((index+EvaluationBatch)%256==0) {
                    scores.flush();std::cout<<"{\"model\":\""<<names[model_index]<<"\",\"samples_evaluated\":"<<index+EvaluationBatch
                        <<",\"elapsed_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-begun).count()<<"}"<<std::endl;
                }
            }
            double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begun).count();
            std::cout<<std::setprecision(17)<<"{\"model\":\""<<names[model_index]<<"\",\"complete\":true,\"last_bpb\":"<<total_nll/std::log(2.)/total_bytes
                <<",\"last_nll\":"<<total_nll<<",\"last_bytes\":"<<total_bytes<<",\"seconds\":"<<elapsed<<"}"<<std::endl;
        }
        std::cout<<"{\"evaluation_complete\":true,\"optimizer_updates\":0,\"peak_arena_gib\":"<<double(arena.peak)/GiB<<"}"<<std::endl;return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 1;}
}
