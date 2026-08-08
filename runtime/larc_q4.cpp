#include "larc_q4.h"
#include <algorithm>
#include <cassert>
#include <cstring>

namespace larc {
float fp16_bits_to_float(std::uint16_t h) {
    const std::uint32_t sign=(std::uint32_t(h&0x8000u))<<16;
    std::uint32_t exp=(h>>10)&0x1Fu;
    std::uint32_t mant=h&0x03FFu;
    std::uint32_t out;
    if(exp==0) {
        if(mant==0) out=sign;
        else {
            int e=-14;
            while((mant&0x0400u)==0){mant<<=1;--e;}
            mant&=0x03FFu;
            out=sign | (std::uint32_t(e+127)<<23) | (mant<<13);
        }
    } else if(exp==31) {
        out=sign|0x7F800000u|(mant<<13);
    } else {
        out=sign|((exp+112u)<<23)|(mant<<13);
    }
    float f;std::memcpy(&f,&out,sizeof(f));return f;
}

std::uint16_t float_to_fp16_bits(float f) {
    std::uint32_t x;std::memcpy(&x,&f,sizeof(x));
    const std::uint16_t sign=std::uint16_t((x>>16)&0x8000u);
    const std::uint32_t mant=x&0x007FFFFFu;
    const int exp=int((x>>23)&0xFFu)-127;
    if(exp>15) return std::uint16_t(sign|0x7C00u);
    if(exp<-24) return sign;
    if(exp<-14) {
        std::uint32_t m=mant|0x00800000u;
        const int shift=(-14-exp);
        std::uint32_t rounded=(m + (1u<<(shift+12)) - 1u + ((m>>(shift+13))&1u))>>(shift+13);
        return std::uint16_t(sign|rounded);
    }
    std::uint32_t he=std::uint32_t(exp+15);
    std::uint32_t hm=(mant+0x00000FFFu+((mant>>13)&1u))>>13;
    if(hm==0x400u){hm=0;++he;if(he>=31)return std::uint16_t(sign|0x7C00u);}
    return std::uint16_t(sign|(he<<10)|hm);
}

static inline int q4_at(const std::uint8_t* row,std::size_t j) {
    const std::uint8_t b=row[j>>1];
    const int code=(j&1)?((b>>4)&0x0F):(b&0x0F);
    return code-8;
}

static inline float q4_row_dot(const Q4Rows& w,std::size_t i,const float* x) {
    const std::size_t stride=(w.cols+1)>>1;
    const std::uint8_t* row=w.packed+i*stride;
    float acc=0.0f;std::size_t j=0;
    for(;j+1<w.cols;j+=2) {
        const std::uint8_t byte=row[j>>1];
        const int q0=int(byte&0x0F)-8;
        const int q1=int((byte>>4)&0x0F)-8;
        acc+=float(q0)*x[j]+float(q1)*x[j+1];
    }
    if(j<w.cols) acc+=float(q4_at(row,j))*x[j];
    return acc*fp16_bits_to_float(w.scales_fp16[i]);
}

void q4_gemv(const Q4Rows& w,const float* x,float* y) {
    for(std::size_t i=0;i<w.rows;++i) y[i]=q4_row_dot(w,i,x);
}

void q4_gemv_add(const Q4Rows& w,const float* x,float* y) {
    for(std::size_t i=0;i<w.rows;++i) y[i]+=q4_row_dot(w,i,x);
}

void q4_transposed_gemv(const Q4Rows& w,const float* x,float* y) {
    std::fill(y,y+w.cols,0.0f);
    const std::size_t stride=(w.cols+1)>>1;
    for(std::size_t i=0;i<w.rows;++i) {
        const float a=x[i]*fp16_bits_to_float(w.scales_fp16[i]);
        const std::uint8_t* row=w.packed+i*stride;
        for(std::size_t j=0;j<w.cols;++j) y[j]+=a*float(q4_at(row,j));
    }
}

void q4_projected_gemv(const Q4Rows& b,const Q4Rows& a,
                       const float* x,float* rank_scratch,float* y) {
    assert(a.cols==b.rows);
    q4_gemv(b,x,rank_scratch);
    q4_gemv(a,rank_scratch,y);
}

void q4_shared_residual_gemv(const Q4Rows& shared,const Q4Rows& b,const Q4Rows& a,
                             const float* x,float* rank_scratch,float* y) {
    assert(shared.cols==b.cols);
    assert(shared.rows==a.rows);
    assert(a.cols==b.rows);
    q4_gemv(shared,x,y);
    q4_gemv(b,x,rank_scratch);
    q4_gemv_add(a,rank_scratch,y);
}
} // namespace larc
