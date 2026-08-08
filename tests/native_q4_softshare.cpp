#include "../runtime/larc_q4.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

struct OwnedQ4 {
    std::vector<std::uint8_t> p; std::vector<std::uint16_t> s; std::size_t rows,cols;
    larc::Q4Rows view() const { return {p.data(),s.data(),rows,cols}; }
};
static OwnedQ4 q4(const std::vector<float>&w,std::size_t rows,std::size_t cols){
    OwnedQ4 o;o.rows=rows;o.cols=cols;o.s.resize(rows);o.p.assign(rows*((cols+1)/2),0x88);const std::size_t stride=(cols+1)/2;
    for(std::size_t i=0;i<rows;i++){
        float pos=0,neg=0;for(std::size_t j=0;j<cols;j++){float v=w[i*cols+j];pos=std::max(pos,v);neg=std::max(neg,-v);}float sf=std::max(1e-12f,std::max(pos/7.0f,neg/8.0f));o.s[i]=larc::float_to_fp16_bits(sf);float s=larc::fp16_bits_to_float(o.s[i]);
        for(std::size_t j=0;j<cols;j++){int qv=int(std::lrint(w[i*cols+j]/s));qv=std::max(-8,std::min(7,qv));std::uint8_t c=std::uint8_t(qv+8);auto&b=o.p[i*stride+(j>>1)];if(j&1)b=(b&0x0F)|(c<<4);else b=(b&0xF0)|c;}
    }return o;
}
static std::vector<float> dq(const OwnedQ4&o){
    std::vector<float>w(o.rows*o.cols);std::size_t st=(o.cols+1)/2;
    for(std::size_t i=0;i<o.rows;i++){float s=larc::fp16_bits_to_float(o.s[i]);for(std::size_t j=0;j<o.cols;j++){auto b=o.p[i*st+(j>>1)];int c=(j&1)?((b>>4)&15):(b&15);w[i*o.cols+j]=float(c-8)*s;}}
    return w;
}
int main(){
    constexpr std::size_t M=173,K=211,R=23;std::mt19937 g(11);std::normal_distribution<float>n(0,0.2f);
    std::vector<float>S(M*K),B(R*K),A(M*R),x(K);for(auto&z:S)z=n(g);for(auto&z:B)z=n(g);for(auto&z:A)z=n(g);for(auto&z:x)z=n(g);
    auto qs=q4(S,M,K),qb=q4(B,R,K),qa=q4(A,M,R);std::vector<float>scratch(R),y(M),ref(M),z(R);larc::q4_shared_residual_gemv(qs.view(),qb.view(),qa.view(),x.data(),scratch.data(),y.data());
    auto sd=dq(qs),bd=dq(qb),ad=dq(qa);
    for(std::size_t r=0;r<R;r++){float v=0;for(std::size_t j=0;j<K;j++)v+=bd[r*K+j]*x[j];z[r]=v;}
    for(std::size_t i=0;i<M;i++){float v=0;for(std::size_t j=0;j<K;j++)v+=sd[i*K+j]*x[j];for(std::size_t r=0;r<R;r++)v+=ad[i*R+r]*z[r];ref[i]=v;}
    float e=0;for(std::size_t i=0;i<M;i++)e=std::max(e,std::fabs(y[i]-ref[i]));
    std::printf("max_abs_error=%.9g rank_scratch_bytes=%zu shared_bytes=%zu residual_factor_bytes=%zu\n",e,R*sizeof(float),larc::q4_storage_bytes(qs.view()),larc::q4_storage_bytes(qb.view())+larc::q4_storage_bytes(qa.view()));
    return e<1e-3f?0:1;
}
