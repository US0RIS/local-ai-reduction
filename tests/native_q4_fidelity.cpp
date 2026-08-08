#include "../runtime/larc_q4.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

struct OQ{std::vector<std::uint8_t>p;std::vector<std::uint16_t>s;std::size_t r,c;larc::Q4Rows v()const{return{p.data(),s.data(),r,c};}};
static OQ q4(const std::vector<float>&w,std::size_t r,std::size_t c){OQ o;o.r=r;o.c=c;o.s.resize(r);o.p.assign(r*((c+1)/2),0x88);auto st=(c+1)/2;for(std::size_t i=0;i<r;i++){float pos=0,neg=0;for(std::size_t j=0;j<c;j++){float v=w[i*c+j];pos=std::max(pos,v);neg=std::max(neg,-v);}float sf=std::max(1e-12f,std::max(pos/7.0f,neg/8.0f));o.s[i]=larc::float_to_fp16_bits(sf);float s=larc::fp16_bits_to_float(o.s[i]);for(std::size_t j=0;j<c;j++){int q=int(std::lrint(w[i*c+j]/s));q=std::max(-8,std::min(7,q));std::uint8_t code=std::uint8_t(q+8);auto&b=o.p[i*st+(j>>1)];if(j&1)b=(b&15)|(code<<4);else b=(b&240)|code;}}return o;}
static void run(float noise){constexpr std::size_t M=1536,K=576,R=32;std::mt19937 g(9);std::normal_distribution<float>nd(0,1);std::vector<float>B(R*K),A(M*R),W(M*K),x(K),yd(M),yl(M),z(R),yfp(M);for(auto&a:B)a=nd(g)/std::sqrt(float(K));for(auto&a:A)a=nd(g)/std::sqrt(float(R));for(std::size_t i=0;i<M;i++)for(std::size_t j=0;j<K;j++){float s=0;for(std::size_t r=0;r<R;r++)s+=A[i*R+r]*B[r*K+j];W[i*K+j]=s+noise*nd(g);}for(auto&a:x)a=nd(g);auto qw=q4(W,M,K),qb=q4(B,R,K),qa=q4(A,M,R);larc::q4_gemv(qw.v(),x.data(),yd.data());larc::q4_projected_gemv(qb.v(),qa.v(),x.data(),z.data(),yl.data());double se=0,sp=0,sefp=0,spfp=0;for(std::size_t i=0;i<M;i++){double exact=0;for(std::size_t j=0;j<K;j++)exact+=double(W[i*K+j])*x[j];yfp[i]=float(exact);double e=double(yd[i])-yl[i];se+=e*e;sp+=double(yd[i])*yd[i];double ef=double(yfp[i])-yl[i];sefp+=ef*ef;spfp+=double(yfp[i])*yfp[i];}double signal_var=1.0/576.0,noise_var=double(noise)*noise,theory=noise_var/(signal_var+noise_var);std::printf("noise=%.6f theory_rank_residual_fraction=%.9f output_nmse_vs_dense_q4=%.9f output_nmse_vs_fp32=%.9f\n",noise,theory,se/sp,sefp/spfp);}
int main(){run(0.02f);run(0.002f);}
