// Separate-storage prompt consumers against the original layout; synthetic screening only.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#include "nvfp4_experts.cu"

static void ck(cudaError_t rc) {
  if (rc != cudaSuccess) { std::fprintf(stderr, "CUDA ERROR: %s\n", cudaGetErrorString(rc)); std::exit(2); }
}
struct Buf {
  void* p; size_t n;
  explicit Buf(size_t bytes): n(bytes) { ck(cudaMalloc(&p, bytes)); }
  ~Buf() { cudaFree(p); }
  void put(const void* x) { ck(cudaMemcpy(p, x, n, cudaMemcpyHostToDevice)); }
  void get(void* x) { ck(cudaMemcpy(x, p, n, cudaMemcpyDeviceToHost)); }
};
struct Variant { const char* name; const void* f; int wn; };
static std::vector<Variant> variants(bool up) {
  if (up) return {{"legacy", (const void*)tf_nvfp4_prompt_swiglu, 4},
    {"separate", (const void*)tf_nvfp4_separate_prompt_up_w4, 4}};
  return {{"legacy", (const void*)tf_nvfp4_prompt_down, 8},
    {"separate", (const void*)tf_nvfp4_separate_prompt_down_w8, 8}};
}
static uint32_t rand32(uint32_t& x) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; }
static int shared(int parts, int wn) {
  int stage = (64 * 8 + wn * 2 * parts * 36) * 16;
  return stage * ((stage * 4 <= 49152) ? 4 : (stage * 3 <= 49152) ? 3 : 2);
}

static void one(bool up, int n, int k, int experts, int count, bool nonfinite, bool timing) {
  const int parts = up ? 2 : 1, pairs = experts * count, slots = up ? 8 : 0;
  const int rows = up ? (pairs + 7) / 8 : pairs;
  const size_t guard = 64;
  uint32_t seed = 1729;
  std::vector<uint32_t> weights(size_t(experts) * (n / 32) * (k / 32) * parts * 144);
  for (size_t i = 0; i < weights.size(); ++i) {
    auto value = rand32(seed);
    weights[i] = (i % 144 < 128) ? value : (0x30303030u | (value & 0x07070707u));
  }
  std::vector<uint16_t> input(size_t(rows) * k + 2 * guard, 0x5a5a);
  for (size_t i = guard; i < input.size() - guard; ++i) {
    float value = (int(rand32(seed) & 65535) - 32768) / 32768.0f;
    __nv_bfloat16 bf = __float2bfloat16_rn(value);
    std::memcpy(&input[i], &bf, 2);
  }
  if (nonfinite) { input[guard] = 0x7fc0; input[guard + 3] = 0x7f80; input[guard + 7] = 0xff80; }
  std::vector<float> scales(experts * parts, 0.02f);
  std::vector<int> items(experts * 3), members(pairs), counts{experts, 0};
  for (int e = 0; e < experts; ++e) {
    items[3*e] = e; items[3*e+1] = e*count; items[3*e+2] = count;
    for (int r = 0; r < count; ++r) members[e*count+r] = r*experts+e;
  }
  Buf w(weights.size()*4), x(input.size()*2), sc(scales.size()*4), it(items.size()*4),
      mem(members.size()*4), cnt(8), out((size_t(pairs)*n + 2*guard)*2);
  w.put(weights.data()); x.put(input.data()); sc.put(scales.data()); it.put(items.data());
  mem.put(members.data()); cnt.put(counts.data());
  std::vector<uint32_t> code(weights.size()/144*128), block_scale(weights.size()/144*16);
  for(size_t i=0;i<weights.size()/144;++i) {
    std::copy_n(weights.data()+i*144,128,code.data()+i*128);
    std::copy_n(weights.data()+i*144+128,16,block_scale.data()+i*16);
  }
  Buf separate_w(code.size()*4), separate_s(block_scale.size()*4);
  separate_w.put(code.data()); separate_s.put(block_scale.data());
  auto bp=(uint4*)separate_s.p;
  auto xp = (__nv_bfloat16*)x.p + guard;
  auto wp = (uint4*)w.p; auto sp = (float*)sc.p; auto ip = (int*)it.p;
  auto cp = (int*)cnt.p; auto mp = (int*)mem.p; auto op = (uint16_t*)out.p + guard;
  int kg = k/32, nb = n/32, stride = k, sl = slots, nn = n, skip = -1;
  float limit = 0;
  void* args[] = {&xp, &stride, &sl, &wp, &sp, &kg, &nb, &ip, &cp, &mp, &op, &nn, &limit, &skip, &bp};
  auto vv = variants(up);
  auto launch = [&](int vi) {
    auto v = vv[vi];
    wp=(uint4*)(vi ? separate_w.p : w.p);
    ck(cudaLaunchKernel(v.f, dim3(experts*((nb+v.wn-1)/v.wn)), dim3(v.wn*32), args, shared(parts,v.wn), 0));
  };
  for (auto v: vv) {
    cudaFuncAttributes a{}; ck(cudaFuncGetAttributes(&a,v.f));
    ck(cudaFuncSetAttribute(v.f,cudaFuncAttributeMaxDynamicSharedMemorySize,shared(parts,v.wn)));
    if (timing && count==1) std::printf("ATTR %s %s threads=%d registers=%d local=%zu shared=%d\n",
      up?"up":"down",v.name,v.wn*32,a.numRegs,a.localSizeBytes,shared(parts,v.wn));
  }
  std::vector<uint16_t> base(out.n/2), got(out.n/2);
  ck(cudaMemset(out.p,0xa5,out.n)); launch(0); ck(cudaDeviceSynchronize()); out.get(base.data());
  for (size_t i = 0; i < guard; ++i) if (base[i]!=0xa5a5 || base[base.size()-1-i]!=0xa5a5) std::exit(3);
  for (size_t vi = 0; vi < vv.size(); ++vi) {
    for (int repeat = 0; repeat < 5; ++repeat) {
      ck(cudaMemset(out.p,0xa5,out.n)); launch(vi); ck(cudaDeviceSynchronize()); out.get(got.data());
      if (got != base) {
        size_t different = 0; for(size_t i=0;i<got.size();++i) different += got[i]!=base[i];
        std::printf("FAIL %s n=%d k=%d count=%d nonfinite=%d variant=%s mismatches=%zu\n",
          up?"up":"down",n,k,count,nonfinite,vv[vi].name,different); std::exit(4);
      }
    }
    std::vector<uint16_t> input_after(input.size()); x.get(input_after.data());
    if (input_after != input) { std::fprintf(stderr,"Input changed\n");std::exit(5); }
  }
  std::vector<uint32_t> weight_after(weights.size()); w.get(weight_after.data());
  if (weight_after != weights) { std::fprintf(stderr,"Weights changed\n");std::exit(6); }
  std::vector<uint32_t> code_after(code.size()), scale_after(block_scale.size());
  separate_w.get(code_after.data()); separate_s.get(scale_after.data());
  if(code_after!=code || scale_after!=block_scale) std::exit(7);
  if (timing) {
    cudaEvent_t a,b; ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
    std::vector<int> order(vv.size());std::iota(order.begin(),order.end(),0);
    std::mt19937 rng(1729);
    for(int round=0;round<5;++round) {
      std::shuffle(order.begin(),order.end(),rng);
      for(auto vi:order) {
        for(int j=0;j<3;++j) launch(vi);
        ck(cudaEventRecord(a)); for(int j=0;j<30;++j) launch(vi); ck(cudaEventRecord(b));
        ck(cudaEventSynchronize(b)); float ms;ck(cudaEventElapsedTime(&ms,a,b));
        std::printf("RESULT %s n=%d k=%d experts=%d count=%d variant=%s wn=%d round=%d ms=%.6f\n",
          up?"up":"down",n,k,experts,count,vv[vi].name,vv[vi].wn,round,ms/30);
      }
    }
    ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));
  }
  std::printf("PASS %s n=%d k=%d experts=%d count=%d nonfinite=%d variants=%zu repeats=5 byte_equal guards inputs\n",
    up?"up":"down",n,k,experts,count,nonfinite,vv.size());std::fflush(stdout);
}
int main() {
  cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0));
  std::printf("DEVICE %s sm=%d%d sms=%d\n",prop.name,prop.major,prop.minor,prop.multiProcessorCount);
  if(prop.major!=12 || prop.minor!=0) return 10;
  for(bool up: {true,false}) {
    for(int n: {32,96}) for(int count:{1,15,17,63,64}) one(up,n,up?4096:1024,3,count,false,false);
    one(up,96,up?4096:1024,3,17,true,false);
    for(int count:{1,8,16,32,64}) one(up,up?1024:4096,up?4096:1024,288,count,false,true);
  }
  std::puts("SUMMARY PASS all template variants: finite/nonfinite replacement equality, repeats, guards and unchanged inputs; timing is isolated synthetic screening only");
}
