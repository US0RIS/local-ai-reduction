#include "../runtime/larc_q4.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

int main(){
    constexpr std::size_t R=7,C=130,G=64;
    const std::size_t stride=(C+1)/2,groups=(C+G-1)/G;
    std::mt19937 gen(123);std::normal_distribution<float> nd(0.f,1.f);
    std::vector<float>w(R*C),x(C),ref(R),got(R);for(auto&v:w)v=nd(gen);for(auto&v:x)v=nd(gen);
    std::vector<std::uint8_t>p(R*stride,0x88);std::vector<std::uint16_t>s(R*groups);
    for(std::size_t i=0;i<R;++i){
        for(std::size_t g=0;g<groups;++g){
            std::size_t b=g*G,e=std::min(C,b+G);float pos=0,neg=0;
            for(std::size_t j=b;j<e;++j){float v=w[i*C+j];pos=std::max(pos,v);neg=std::max(neg,-v);}
            float sf=std::max(1e-8f,std::max(pos/7.f,neg/8.f));s[i*groups+g]=larc::float_to_fp16_bits(sf);float sh=larc::fp16_bits_to_float(s[i*groups+g]);
            for(std::size_t j=b;j<e;++j){int q=int(std::lrint(w[i*C+j]/sh));q=std::max(-8,std::min(7,q));std::uint8_t code=std::uint8_t(q+8);auto&byte=p[i*stride+(j>>1)];if(j&1)byte=(byte&0x0F)|(code<<4);else byte=(byte&0xF0)|code;}
        }
    }
    for(std::size_t i=0;i<R;++i){double a=0;for(std::size_t j=0;j<C;++j){std::uint8_t byte=p[i*stride+(j>>1)];int q=int((j&1)?((byte>>4)&15):(byte&15))-8;float sh=larc::fp16_bits_to_float(s[i*groups+j/G]);a+=double(q)*sh*x[j];}ref[i]=float(a);}
    larc::Q4GroupRows q{p.data(),s.data(),R,C,G};larc::q4_grouped_gemv(q,x.data(),got.data());
    float mx=0;for(std::size_t i=0;i<R;++i)mx=std::max(mx,std::abs(ref[i]-got[i]));
    const std::size_t expected=R*stride+R*groups*sizeof(std::uint16_t);
    std::printf("max_abs_error=%.9g storage_bytes=%zu expected_bytes=%zu groups=%zu\n",mx,larc::q4_grouped_storage_bytes(q),expected,groups);
    return (mx<2e-5f && larc::q4_grouped_storage_bytes(q)==expected)?0:1;
}
