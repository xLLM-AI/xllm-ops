#pragma once
#include <pto/pto-inst.hpp>
#include <type_traits>

namespace kda_prefill {
#if defined(__DAV_C310_VEC__)
// Keep the rounded product in a register until TwoSum consumes it. Use a
// separate vmul (not vmula): both the original product rounding and every
// compensated-add operation must remain identical to the staged path.
template <typename Matrix, typename Row>
__tf__ PTO_INTERNAL void MulAddCompensatedRowsA5(
    typename Matrix::TileDType __out__ sum_out,
    typename Matrix::TileDType __out__ low_out,
    typename Matrix::TileDType __in__ sum_in,
    typename Matrix::TileDType __in__ low_in,
    typename Matrix::TileDType __in__ input,
    typename Row::TileDType __in__ weight, uint16_t rows)
{
    using namespace pto;
    auto* sp=(__ubuf__ float*)__cce_get_tile_ptr(sum_in);
    auto* lp=(__ubuf__ float*)__cce_get_tile_ptr(low_in);
    auto* xp=(__ubuf__ float*)__cce_get_tile_ptr(input);
    auto* wp=(__ubuf__ float*)__cce_get_tile_ptr(weight);
    auto* so=(__ubuf__ float*)__cce_get_tile_ptr(sum_out);
    auto* lo=(__ubuf__ float*)__cce_get_tile_ptr(low_out);
    __VEC_SCOPE__ {
        RegTensor<float> a,b,s,v,e,t,l,x,w;
        MaskReg mask=PSetWithType<float>(PAT_ALL);
        constexpr auto dist=std::integral_constant<::DistVST,
            static_cast<::DistVST>(GetDistVst<float,pto::DistVST::DIST_NORM>())>();
        for(uint16_t half=0;half<2;++half) {
            vlds(w,wp,half*64,NORM);
            for(uint16_t row=0;row<rows;++row) {
                const uint32_t offset=row*128+half*64;
                vlds(x,xp,offset,NORM);vmul(b,x,w,mask,MODE_ZEROING);
                vlds(a,sp,offset,NORM);vlds(l,lp,offset,NORM);
                vadd(s,a,b,mask,MODE_ZEROING);
                vsub(v,s,a,mask,MODE_ZEROING);
                vsub(t,s,v,mask,MODE_ZEROING);
                vsub(e,a,t,mask,MODE_ZEROING);
                vsub(t,b,v,mask,MODE_ZEROING);
                vadd(e,e,t,mask,MODE_ZEROING);vadd(l,l,e,mask,MODE_ZEROING);
                vsts(s,so,offset,dist,mask);vsts(l,lo,offset,dist,mask);
            }
        }
    }
}

// Small-argument SiLU avoids exp/div rounding near BF16 midpoints. Outside
// |x| < 0.5 retain the existing exp/div formula. All intermediates stay FP32.
template <typename Matrix>
__tf__ PTO_INTERNAL void SiluRowsA5(typename Matrix::TileDType __out__ output,
                                   typename Matrix::TileDType __in__ input,
                                   typename Matrix::TileDType __in__ low_input,
                                   uint16_t rows)
{
    using namespace pto;
    auto* in = (__ubuf__ float*)__cce_get_tile_ptr(input);
    auto* out = (__ubuf__ float*)__cce_get_tile_ptr(output);
    auto* low = (__ubuf__ float*)__cce_get_tile_ptr(low_input);
    __VEC_SCOPE__ {
        RegTensor<float> x,z,p,c,half_x,approx,regular,absolute;
        RegTensor<float> zerr,z2,y,sum,b,err,tmp,lo,derivative;
        MaskReg mask=PSetWithType<float>(PAT_ALL),small;
        constexpr auto dist=std::integral_constant<::DistVST,
            static_cast<::DistVST>(GetDistVst<float,pto::DistVST::DIST_NORM>())>();
        for(uint16_t block=0;block<rows*2;++block) {
            vlds(x,in,block*64,NORM);
            vlds(lo,low,block*64,NORM);
            // Cancellation can leave the carried residual larger than x.
            // Renormalize with full TwoSum (not FastTwoSum) before selecting
            // the SiLU interval; retain the new low part for compensation.
            vadd(sum,x,lo,mask,MODE_ZEROING);
            vsub(b,sum,x,mask,MODE_ZEROING);
            vsub(tmp,sum,b,mask,MODE_ZEROING);
            vsub(err,x,tmp,mask,MODE_ZEROING);
            vsub(tmp,lo,b,mask,MODE_ZEROING);
            vadd(lo,err,tmp,mask,MODE_ZEROING);
            vadds(x,sum,0.0f,mask,MODE_ZEROING);
            vmul(z,x,x,mask,MODE_ZEROING);
            vbr(p,-691.0f/319334400.0f);
            vbr(c,31.0f/1451520.0f); vmadd(p,z,c,mask,MODE_ZEROING);
            vbr(c,-17.0f/80640.0f); vmadd(p,z,c,mask,MODE_ZEROING);
            vbr(c,1.0f/480.0f); vmadd(p,z,c,mask,MODE_ZEROING);
            vbr(c,-1.0f/48.0f); vmadd(p,z,c,mask,MODE_ZEROING);
            vmuls(half_x,x,0.5f,mask,MODE_ZEROING);
            // TwoSum keeps the low part of x/2 + x*x/4 until the small
            // higher-order correction has been added; avoid double rounding.
            vmuls(zerr,z,-1.0f,mask,MODE_ZEROING);
            vmula(zerr,x,x,mask,MODE_ZEROING);
            vmuls(y,z,0.25f,mask,MODE_ZEROING);
            vadd(sum,half_x,y,mask,MODE_ZEROING);
            vsub(b,sum,half_x,mask,MODE_ZEROING);
            vsub(tmp,sum,b,mask,MODE_ZEROING);
            vsub(err,half_x,tmp,mask,MODE_ZEROING);
            vsub(tmp,y,b,mask,MODE_ZEROING);
            vadd(err,err,tmp,mask,MODE_ZEROING);
            vmuls(zerr,zerr,0.25f,mask,MODE_ZEROING);
            vadd(err,err,zerr,mask,MODE_ZEROING);
            vmul(z2,z,z,mask,MODE_ZEROING);
            vmadd(p,z2,err,mask,MODE_ZEROING);
            vbr(derivative,-691.0f/26611200.0f);
            vbr(c,31.0f/145152.0f);vmadd(derivative,z,c,mask,MODE_ZEROING);
            vbr(c,-17.0f/10080.0f);vmadd(derivative,z,c,mask,MODE_ZEROING);
            vbr(c,1.0f/80.0f);vmadd(derivative,z,c,mask,MODE_ZEROING);
            vbr(c,-1.0f/12.0f);vmadd(derivative,z,c,mask,MODE_ZEROING);
            vbr(c,0.5f);vmadd(derivative,z,c,mask,MODE_ZEROING);
            vmadd(derivative,x,c,mask,MODE_ZEROING);
            vmula(p,lo,derivative,mask,MODE_ZEROING);
            vadd(p,sum,p,mask,MODE_ZEROING);
            vmuls(regular,x,-1.0f,mask,MODE_ZEROING);
            vexp(regular,regular,mask,MODE_ZEROING);
            vadds(regular,regular,1.0f,mask,MODE_ZEROING);
            vdiv(regular,x,regular,mask,MODE_ZEROING);
            vabs(absolute,x,mask,MODE_ZEROING);
            vcmps_lt(small,absolute,0.5f,mask);
            vsel(approx,p,regular,small);
            vsts(approx,out,block*64,dist,mask);
        }
    }
}
// Callers supply absmax before any potentially overflowing square is formed.
// Ordinary rows keep the input bits and original SplitSquaresA5 arithmetic.
// For absmax > 2^60, scale by 2^-68 before any square is formed. For every
// finite BF16 input, 128 * (max(abs(x)) * 2^-68)^2 is then below 2^127.
// Epsilon in scaled coordinates is 1e-6 * 2^-136, which rounds to FP32 zero.
// scaled_out may alias input; ordinary rows are never written through it.
template <typename Matrix, typename Column>
__tf__ PTO_INTERNAL void SplitSquaresRangeA5(
    typename Matrix::TileDType __out__ high_out,
    typename Matrix::TileDType __out__ low_out,
    typename Matrix::TileDType __out__ scaled_out,
    typename Column::TileDType __out__ epsilon_out,
    typename Matrix::TileDType __in__ input,
    typename Column::TileDType __in__ abs_maximum, uint16_t rows)
{
    using namespace pto;
    auto* hp=(__ubuf__ float*)__cce_get_tile_ptr(high_out);
    auto* lp=(__ubuf__ float*)__cce_get_tile_ptr(low_out);
    auto* xp=(__ubuf__ float*)__cce_get_tile_ptr(input);
    auto* op=(__ubuf__ float*)__cce_get_tile_ptr(scaled_out);
    auto* ep=(__ubuf__ float*)__cce_get_tile_ptr(epsilon_out);
    auto* mp=(__ubuf__ float*)__cce_get_tile_ptr(abs_maximum);
    __VEC_SCOPE__ {
        RegTensor<float> x,p,m,scaled,hi,lo,eps,zero;
        RegTensor<uint32_t> exponent,sbits,ibits,constant;
        MaskReg mask=PSetWithType<float>(PAT_ALL),large;
        uint32_t row_count=rows;
        MaskReg row_mask=CreatePredicate<float>(row_count);
        constexpr auto dist=std::integral_constant<::DistVST,
            static_cast<::DistVST>(GetDistVst<float,pto::DistVST::DIST_NORM>())>();
        vlds(m,mp,0,NORM);
        vcmps_gt(large,m,0x1p60f,row_mask);
        vbr(eps,1e-6f);vbr(zero,0.0f);
        vsel(eps,zero,eps,large);
        vsts(eps,ep,0,dist,row_mask);
        for(uint16_t row=0;row<rows;++row) {
            vlds(m,mp,row,BRC_B32);
            vcmps_gt(large,m,0x1p60f,mask);
            vmuls(scaled,m,0x1p-68f,mask,MODE_ZEROING);
            vsel(m,scaled,m,large);
            vmul(m,m,m,mask,MODE_ZEROING);
            vmaxs(m,m,0x1p-24f,mask,MODE_ZEROING);
            vshrs(exponent,(RegTensor<uint32_t>&)m,23,mask,MODE_ZEROING);
            vbr(constant,uint32_t(269));vsub(sbits,constant,exponent,mask,MODE_ZEROING);
            vbr(constant,uint32_t(15));vsub(ibits,exponent,constant,mask,MODE_ZEROING);
            vshls(sbits,sbits,23,mask,MODE_ZEROING);vshls(ibits,ibits,23,mask,MODE_ZEROING);
            for(uint16_t half=0;half<2;++half) {
                const uint32_t offset=row*128+half*64;
                vlds(x,xp,offset,NORM);
                vmuls(scaled,x,0x1p-68f,mask,MODE_ZEROING);
                vsel(x,scaled,x,large);
                vsts(x,op,offset,dist,large);
                vmul(p,x,x,mask,MODE_ZEROING);
                vmul(scaled,p,(RegTensor<float>&)sbits,mask,MODE_ZEROING);
                vadds(hi,scaled,8388608.0f,mask,MODE_ZEROING);
                vadds(hi,hi,-8388608.0f,mask,MODE_ZEROING);
                vsub(lo,scaled,hi,mask,MODE_ZEROING);
                vmul(hi,hi,(RegTensor<float>&)ibits,mask,MODE_ZEROING);
                vmul(lo,lo,(RegTensor<float>&)ibits,mask,MODE_ZEROING);
                vsts(hi,hp,offset,dist,mask);vsts(lo,lp,offset,dist,mask);
            }
        }
    }
}

// Same residual correction as NormDenominatorA5; only epsilon acquisition differs.
template <typename Column>
__tf__ PTO_INTERNAL void NormDenominatorWithEpsilonA5(
    typename Column::TileDType __out__ high_out,
    typename Column::TileDType __out__ low_out,
    typename Column::TileDType __in__ high_in,
    typename Column::TileDType __in__ low_in,
    typename Column::TileDType __in__ epsilon_in, uint16_t rows)
{
    using namespace pto;
    auto* hi=(__ubuf__ float*)__cce_get_tile_ptr(high_in);
    auto* lo=(__ubuf__ float*)__cce_get_tile_ptr(low_in);
    auto* ho=(__ubuf__ float*)__cce_get_tile_ptr(high_out);
    auto* lout=(__ubuf__ float*)__cce_get_tile_ptr(low_out);
    auto* ep=(__ubuf__ float*)__cce_get_tile_ptr(epsilon_in);
    __VEC_SCOPE__ {
        RegTensor<float> h,l,s,b,e,tmp,d,neg,eps;
        uint32_t count=rows;MaskReg mask=CreatePredicate<float>(count);
        constexpr auto dist=std::integral_constant<::DistVST,
            static_cast<::DistVST>(GetDistVst<float,pto::DistVST::DIST_NORM>())>();
        vlds(h,hi,0,NORM);vlds(l,lo,0,NORM);vlds(eps,ep,0,NORM);
        vadd(s,h,eps,mask,MODE_ZEROING);vsub(b,s,h,mask,MODE_ZEROING);
        vsub(tmp,s,b,mask,MODE_ZEROING);vsub(e,h,tmp,mask,MODE_ZEROING);
        vsub(tmp,eps,b,mask,MODE_ZEROING);vadd(e,e,tmp,mask,MODE_ZEROING);
        vadd(l,l,e,mask,MODE_ZEROING);vsqrt(d,s,mask,MODE_ZEROING);
        vmuls(neg,d,-1.0f,mask,MODE_ZEROING);vmula(s,neg,d,mask,MODE_ZEROING);
        vadd(s,s,l,mask,MODE_ZEROING);vmuls(tmp,d,2.0f,mask,MODE_ZEROING);
        vdiv(e,s,tmp,mask,MODE_ZEROING);
        vsts(d,ho,0,dist,mask);vsts(e,lout,0,dist,mask);
    }
}

template <typename Matrix, typename Column>
__tf__ PTO_INTERNAL void NormalizeSplitA5(
    typename Matrix::TileDType __out__ output,
    typename Matrix::TileDType __in__ input,
    typename Column::TileDType __in__ denominator,
    typename Column::TileDType __in__ correction, uint16_t rows)
{
    using namespace pto;
    auto* in=(__ubuf__ float*)__cce_get_tile_ptr(input);
    auto* out=(__ubuf__ float*)__cce_get_tile_ptr(output);
    auto* dp=(__ubuf__ float*)__cce_get_tile_ptr(denominator);
    auto* ep=(__ubuf__ float*)__cce_get_tile_ptr(correction);
    __VEC_SCOPE__ {
        RegTensor<float> x,d,de,q,neg,r;
        MaskReg mask=PSetWithType<float>(PAT_ALL);
        constexpr auto dist=std::integral_constant<::DistVST,
            static_cast<::DistVST>(GetDistVst<float,pto::DistVST::DIST_NORM>())>();
        for(uint16_t row=0;row<rows;++row) {
            vlds(d,dp,row,BRC_B32);vlds(de,ep,row,BRC_B32);
            for(uint16_t half=0;half<2;++half) {
                const uint32_t offset=row*128+half*64;
                vlds(x,in,offset,NORM);vdiv(q,x,d,mask,MODE_ZEROING);
                vmuls(neg,q,-1.0f,mask,MODE_ZEROING);vadds(r,x,0.0f,mask,MODE_ZEROING);
                vmula(r,neg,d,mask,MODE_ZEROING);vmula(r,neg,de,mask,MODE_ZEROING);
                vdiv(r,r,d,mask,MODE_ZEROING);vadd(q,q,r,mask,MODE_ZEROING);
                vsts(q,out,offset,dist,mask);
            }
        }
    }
}
#endif
} // namespace kda_prefill
