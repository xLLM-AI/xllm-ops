#pragma once
#include "kda_prefill_inverse_fp32_a5.h"
#include "kda_prefill_chunk_h_a5.h"

namespace mega_kda_prefill_inverse_block_a5 {
#if __CCE_AICORE__ == 310
using namespace pto;
using namespace mega_kda_prefill_utils;
namespace H = mega_kda_prefill_chunk_h_kda;
namespace I = mega_kda_prefill_inverse_fp32;
using GS = Shape<1,1,1,DYNAMIC,DYNAMIC>;
using ST = pto::Stride<1,1,1,DYNAMIC,1>;
using GM = GlobalTensor<float,GS,ST>;

AICORE PTO_INLINE void Resolve(int64_t task, __gm__ int32_t* cu, int batches,
    int heads, int64_t& start, int& valid, int& head) {
  head = task % heads;
  int64_t chunk = task / heads;
  for (int b=0; b<batches; ++b) {
    const int64_t len = cu[b+1]-cu[b], count = (len+127)/128;
    if (chunk < count) {
      start = cu[b] + chunk*128;
      valid = len-chunk*128 < 128 ? len-chunk*128 : 128;
      return;
    }
    chunk -= count;
  }
}

// [A 0; B D]^-1 = [A^-1 0; -D^-1 B A^-1 D^-1].
// Diagonals retain the FP32 row solver; both off-diagonal products use FP32.
AICORE PTO_INLINE void Run(__gm__ float* lower, __gm__ float* out,
    __gm__ float* identity, __gm__ float* scratch, __gm__ int32_t* cu,
    int batches, int heads) {
  int64_t chunks=0;
  for (int b=0;b<batches;++b) chunks += (cu[b+1]-cu[b]+127)/128;
  const int64_t tasks=chunks*heads;
#if defined(__DAV_C310_VEC__)
  set_mask_norm(); set_vector_mask(-1,-1);
  const int lane=get_block_idx()*2+get_subblockid(), lanes=get_block_num()*2;
  I::Vec<64,64> eye, x;
  I::Vec<64,64> reduction;
  TASSIGN(eye,147456); TASSIGN(x,65536); TASSIGN(reduction,131072);
  GM ig(identity,GS(1,1,1,64,64),ST(128)); TLOAD(eye,ig);
  for(int64_t task=lane;task<tasks*2;task+=lanes) {
    const int first=(task/tasks)*64;
    int64_t start=0; int valid=0,head=0;
    Resolve(task%tasks,cu,batches,heads,start,valid,head);
    if(first>=valid) continue;
    const int n=valid-first<64?valid-first:64;
    Tile<TileType::Vec,float,64,128,BLayout::RowMajor,DYNAMIC,64,
         SLayout::NoneBox,512> low(n);
    TASSIGN(low,0);
    GM lg(lower+((start+first)*heads+head)*128+first,
          GS(1,1,1,n,64),ST(heads*128));
    TLOAD(low,lg); SetWaitFlag<PIPE_MTE2,PIPE_V>(0);
    TMOV(x,eye);
    for(uint16_t row=1;row<n;++row) {
      I::InverseRowsA5<decltype(low),decltype(x),decltype(reduction)>(
          x.data(),x.data(),low.data(),eye.data(),reduction.data(),row,0);
    }
    I::Vec<64,64,DYNAMIC> store(n); TASSIGN(store,65536);
    GM dst(out+((start+first)*heads+head)*128+first,
           GS(1,1,1,n,64),ST(heads*128));
    SetWaitFlag<PIPE_V,PIPE_MTE3>(0); TSTORE(dst,store);
    if(first==0) {
      SetWaitFlag<PIPE_MTE3,PIPE_V>(0); TEXPANDS(store,0.0f);
      GM upper(out+(start*heads+head)*128+64,
               GS(1,1,1,n,64),ST(heads*128));
      SetWaitFlag<PIPE_V,PIPE_MTE3>(0); TSTORE(upper,store);
    }
    pipe_barrier(PIPE_ALL);
  }
#endif
  SyncAllMixA5();
  const int core=get_block_idx(), cores=get_block_num();
  auto* temp=scratch+int64_t(core)*5*16384;
  for(int64_t task=core;task<tasks;task+=cores) {
    int64_t start=0; int valid=0,head=0;
    Resolve(task,cu,batches,heads,start,valid,head);
    if(valid<=64) continue;
    const int n=valid-64;
    const int64_t top=(start*heads+head)*128;
    const int64_t bottom=((start+64)*heads+head)*128;
#if defined(__DAV_CUBE__)
    H::TileMatL1<float,64,64> a,b;
    H::DynMatL1<float,64,64> aload(n,64),bload(n,64);
    TileAcc<float,64,64> c;
    H::DynAccTile<float,64,64> result(n,64);
    TASSIGN(a,0); TASSIGN(aload,0); TASSIGN(b,16384); TASSIGN(bload,16384);
    TASSIGN(c,0); TASSIGN(result,0);
    GM ag(lower+bottom,GS(1,1,1,n,64),ST(heads*128));
    GM bg(out+top,GS(1,1,1,64,64),ST(heads*128));
    TLOAD(aload,ag); if(n<64) TFILLPAD(aload,aload);
    TLOAD(b,bg);
    H::gemm_v0<float,float,64,64,64,64,64,64,64,false,false>(a,b,c,true);
    GM pg(temp,GS(1,1,1,n,64),ST(64));
    TSTORE(pg,result); pipe_barrier(PIPE_ALL);
    GM dg(out+bottom+64,GS(1,1,1,n,64),ST(heads*128));
    TLOAD(aload,dg); if(n<64) TFILLPAD(aload,aload);
    TLOAD(bload,pg); if(n<64) TFILLPAD(bload,bload);
    H::gemm_v0<float,float,64,64,64,64,64,64,64,false,false>(a,b,c,true);
    GM yg(temp+4096,GS(1,1,1,n,64),ST(64));
    TSTORE(yg,result); pipe_barrier(PIPE_ALL);
    SignalBothVecOnA5<PIPE_FIX>(12);
    WaitBothVecOnA5<PIPE_MTE2>(10);
#elif defined(__DAV_C310_VEC__)
    wait_intra_block(PIPE_MTE2,12);
    if(get_subblockid()==0) {
      I::Vec<64,64,DYNAMIC> result(n); TASSIGN(result,0);
      GM yg(temp+4096,GS(1,1,1,n,64),ST(64)); TLOAD(result,yg);
      SetWaitFlag<PIPE_MTE2,PIPE_V>(0); TMULS(result,result,-1.0f);
      GM dst(out+bottom,GS(1,1,1,n,64),ST(heads*128));
      SetWaitFlag<PIPE_V,PIPE_MTE3>(0); TSTORE(dst,result);
    }
    set_intra_block(PIPE_MTE3,10);
    pipe_barrier(PIPE_ALL);
#endif
  }
}
#endif
} // namespace mega_kda_prefill_inverse_block_a5
