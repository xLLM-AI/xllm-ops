// SigmoidMul fused-op benchmark & accuracy check.
//   - accuracy (absolute): fused out vs an fp64 CPU golden of x*sigmoid(gate),
//     bf16 tolerance rtol=1/64 atol=1/256.
//   - accuracy (relative): fused out vs unfused aclnnSigmoid + aclnnMul result.
//   - perf: fused aclnnSigmoidMul vs unfused aclnnSigmoid + aclnnMul (re-plan/call).
// Build (example):
//   OPP=<opp>/vendors/<vendor>; A=$ASCEND_HOME_PATH/aarch64-linux
//   g++ -O2 -std=c++17 sigmoid_mul_bench.cpp -o sigmoid_mul_bench \
//     -I$OPP/op_api/include -I$A/include -L$A/lib64 -L$OPP/op_api/lib \
//     -lcust_opapi -lopapi -lnnopbase -lascendcl -lpthread
// Run: LD_LIBRARY_PATH=$OPP/op_api/lib:$A/lib64 ./sigmoid_mul_bench

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
  const double RTOL=1.0/64, ATOL=1.0/256;
  struct C{const char*n;long M;bool bc;};
  C cs[]={{"广播 [4,2048]",4,true},{"广播 [16,2048]",16,true},{"广播 [32,2048]",32,true},
          {"逐元素 [4,2048]",4,false},{"逐元素 [8,2048]",8,false},{"逐元素 [16,2048]",16,false},{"逐元素 [32,2048]",32,false}};
  printf("%-16s %9s %8s %7s | %-24s | %-24s\n","case","unfused","fused","speedup","acc vs unfused","acc vs fp64 golden");
  printf("---------------------------------------------------------------------------------------------------------------\n");
  for(auto&c:cs){
    long M=c.M,N=2048; std::vector<int64_t> xs{M,N}, gs=c.bc?std::vector<int64_t>{M,1}:std::vector<int64_t>{M,N};
    std::vector<uint16_t> hx(M*N),hg(c.bc?M:M*N);
    for(long i=0;i<M*N;i++)hx[i]=f2bf(std::sin(0.013f*i)*3.0f);
    for(long i=0;i<(long)hg.size();i++)hg[i]=f2bf(std::cos(0.021f*i)*2.0f);
    D x=alloc(xs),g=alloc(gs),ref=alloc(xs),out=alloc(xs),sig=alloc(gs);
    aclrtMemcpy(x.p,x.b,hx.data(),x.b,ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(g.p,g.b,hg.data(),g.b,ACL_MEMCPY_HOST_TO_DEVICE);
    sig_mul(x.t,g.t,ref.t,sig.t,s,wb); sigmul(x.t,g.t,out.t,s,wb); aclrtSynchronizeStream(s);
    std::vector<uint16_t> href(M*N),hout(M*N);
    aclrtMemcpy(href.data(),ref.b,ref.p,ref.b,ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(hout.data(),out.b,out.p,out.b,ACL_MEMCPY_DEVICE_TO_HOST);
    // (1) vs unfused
    double m1=0,x1=0; long b1=0;
    // (2) vs fp64 golden
    double m2=0,x2=0; long b2=0;
    for(long r=0;r<M;r++)for(long cc=0;cc<N;cc++){
      long idx=r*N+cc; double gv=bf2f(hg[c.bc?r:idx]); double xv=bf2f(hx[idx]);
      double gold=(double)xv*(1.0/(1.0+std::exp(-gv)));
      double refv=bf2f(href[idx]), outv=bf2f(hout[idx]);
      double relu=std::fabs(refv-outv)/(std::fabs(refv)+1e-6); m1+=relu; x1=std::max(x1,relu); if(relu>0.02)b1++;
      double reld=std::fabs(gold-outv)/(std::fabs(gold)+1e-6); m2+=reld; x2=std::max(x2,reld);
      if(std::fabs(gold-outv) > ATOL+RTOL*std::fabs(gold)) b2++;
    }
    double tot=M*N; m1/=tot;m2/=tot;
    double tu=timed([&]{sig_mul(x.t,g.t,ref.t,sig.t,s,wb);},s,200,20,5);
    double tf=timed([&]{sigmul(x.t,g.t,out.t,s,wb);},s,200,20,5);
    char a1[40],a2[40];
    snprintf(a1,sizeof(a1),"MERE=%.1e MARE=%.1e bad=%ld",m1,x1,b1);
    snprintf(a2,sizeof(a2),"MERE=%.1e MARE=%.1e bad=%ld",m2,x2,b2);
    printf("%-16s %8.2fus %7.2fus %6.2fx | %-24s | %-24s\n",c.n,tu,tf,tu/tf,a1,a2);
  }
  aclrtDestroyStream(s);aclrtResetDevice(0);aclFinalize();return 0;
}
