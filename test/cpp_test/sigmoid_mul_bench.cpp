// SigmoidMul fused-op benchmark & accuracy check.
// Baseline = unfused aclnnSigmoid + aclnnMul (both re-plan per aclnn call).
//   - accuracy: fused aclnnSigmoidMul out vs unfused (aclnnSigmoid+aclnnMul) result (bf16)
//   - perf: fused aclnnSigmoidMul vs unfused pair
// Build (example):
//   OPP=<opp>/vendors/<vendor>; A=$ASCEND_HOME_PATH/aarch64-linux
//   g++ -O2 -std=c++17 sigmoid_mul_bench.cpp -o sigmoid_mul_bench \
//     -I$OPP/op_api/include -I$A/include -L$A/lib64 -L$OPP/op_api/lib \
//     -lcust_opapi -lopapi -lnnopbase -lascendcl -lpthread
// Run:
//   LD_LIBRARY_PATH=$OPP/op_api/lib:$A/lib64 ./sigmoid_mul_bench

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <acl/acl.h>
#include "aclnnop/aclnn_sigmoid.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnn_sigmoid_mul.h"
#define CK(x) do{ if((x)!=ACL_SUCCESS){printf("ACL err %d @%d\n",(int)(x),__LINE__);return 1;} }while(0)
static uint16_t f2bf(float f){uint32_t b;memcpy(&b,&f,4);return (uint16_t)(b>>16);}
static float bf2f(uint16_t v){uint32_t b=(uint32_t)v<<16;float f;memcpy(&f,&b,4);return f;}
static aclTensor* mk(std::vector<int64_t> sh, void* p){std::vector<int64_t> st(sh.size(),1);for(int i=(int)sh.size()-2;i>=0;i--)st[i]=sh[i+1]*st[i+1];return aclCreateTensor(sh.data(),sh.size(),ACL_BF16,st.data(),0,ACL_FORMAT_ND,sh.data(),sh.size(),p);}
struct D{aclTensor* t;void* p;uint64_t b;};
static D alloc(std::vector<int64_t> sh){D r;int64_t n=1;for(auto d:sh)n*=d;r.b=n*2;aclrtMalloc(&r.p,r.b,ACL_MEM_MALLOC_NORMAL_ONLY);r.t=mk(sh,r.p);return r;}
static void sigmul(aclTensor*x,aclTensor*g,aclTensor*out,aclrtStream s,void*wb){uint64_t w=0;aclOpExecutor*e=nullptr;aclnnSigmoidMulGetWorkspaceSize(x,g,out,&w,&e);aclnnSigmoidMul(wb,w,e,s);}
static void sig_mul(aclTensor*x,aclTensor*g,aclTensor*out,aclTensor*sig,aclrtStream s,void*wb){uint64_t w=0;aclOpExecutor*e=nullptr;aclnnSigmoidGetWorkspaceSize(g,sig,&w,&e);aclnnSigmoid(wb,w,e,s);aclnnMulGetWorkspaceSize(x,sig,out,&w,&e);aclnnMul(wb,w,e,s);}
template<class F> static double timed(F f,aclrtStream s,int it,int warm,int rep){for(int i=0;i<warm;i++)f();aclrtSynchronizeStream(s);double best=1e18;for(int r=0;r<rep;r++){aclrtEvent a,b;aclrtCreateEvent(&a);aclrtCreateEvent(&b);aclrtSynchronizeStream(s);aclrtRecordEvent(a,s);for(int i=0;i<it;i++)f();aclrtRecordEvent(b,s);aclrtSynchronizeStream(s);float ms;aclrtEventElapsedTime(&ms,a,b);best=std::min(best,(double)ms*1000/it);aclrtDestroyEvent(a);aclrtDestroyEvent(b);}return best;}
int main(){
  CK(aclInit(nullptr));CK(aclrtSetDevice(0));aclrtStream s=nullptr;CK(aclrtCreateStream(&s));void*wb=nullptr;CK(aclrtMalloc(&wb,8u<<20,ACL_MEM_MALLOC_NORMAL_ONLY));
  struct C{const char*n;long M;bool bc;};
  C cs[]={{"广播 [4,2048]",4,true},{"广播 [16,2048]",16,true},{"广播 [32,2048]",32,true},
          {"逐元素 [4,2048]",4,false},{"逐元素 [8,2048]",8,false},{"逐元素 [16,2048]",16,false},{"逐元素 [32,2048]",32,false}};
  printf("%-16s %10s %10s %8s | %-22s\n","case","unfused","fused","speedup","accuracy(vs unfused)");
  printf("----------------------------------------------------------------------------------------\n");
  for(auto&c:cs){
    long M=c.M,N=2048; std::vector<int64_t> xs{M,N}, gs=c.bc?std::vector<int64_t>{M,1}:std::vector<int64_t>{M,N};
    std::vector<uint16_t> hx(M*N),hg((c.bc?M:M*N));
    for(long i=0;i<M*N;i++)hx[i]=f2bf(std::sin(0.013f*i)*3.0f);
    for(long i=0;i<(long)hg.size();i++)hg[i]=f2bf(std::cos(0.021f*i)*2.0f);
    D x=alloc(xs),g=alloc(gs),ref=alloc(xs),out=alloc(xs),sig=alloc(gs);
    aclrtMemcpy(x.p,x.b,hx.data(),x.b,ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(g.p,g.b,hg.data(),g.b,ACL_MEMCPY_HOST_TO_DEVICE);
    // reference = unfused aclnnSigmoid + aclnnMul
    sig_mul(x.t,g.t,ref.t,sig.t,s,wb); aclrtSynchronizeStream(s);
    sigmul(x.t,g.t,out.t,s,wb); aclrtSynchronizeStream(s);
    std::vector<uint16_t> href(M*N),hout(M*N);
    aclrtMemcpy(href.data(),ref.b,ref.p,ref.b,ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(hout.data(),out.b,out.p,out.b,ACL_MEMCPY_DEVICE_TO_HOST);
    double mere=0,mare=0; long bad=0;
    for(long i=0;i<M*N;i++){double gv=bf2f(href[i]),ov=bf2f(hout[i]);double rel=std::fabs(gv-ov)/(std::fabs(gv)+1e-6);mere+=rel;mare=std::max(mare,rel);if(rel>0.02)bad++;}
    mere/=M*N;
    double tu=timed([&]{sig_mul(x.t,g.t,ref.t,sig.t,s,wb);},s,200,20,5);
    double tf=timed([&]{sigmul(x.t,g.t,out.t,s,wb);},s,200,20,5);
    char acc[48]; snprintf(acc,sizeof(acc),"MERE=%.1e MARE=%.1e bad=%ld/%ld",mere,mare,bad,M*N);
    printf("%-16s %9.2fus %9.2fus %7.2fx | %s\n",c.n,tu,tf,tu/tf,acc);
  }
  aclrtDestroyStream(s);aclrtResetDevice(0);aclFinalize();return 0;
}
