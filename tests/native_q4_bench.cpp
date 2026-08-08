#include "../runtime/larc_q4.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

struct OQ { std::vector<std::uint8_t>p;std::vector<float>s;std::size_t r,c;larc::Q4Rows v()const{return{p.data(),s.data(),r,c};}};
static OQ q4(const std::vector<float>&w,std::size_t r,std::size_t c){OQ o;o.r=r;o.c=c;o.s.resize(r);o.p.assign(r*((c+1)/2),0x88);auto st=(c+1)/2;for(std::size_t i=0;i<r;i++){float m=1e-12f;for(std::size_t j=0;j<c;j++)m=std::max(m,std::fabs(w[i*c+j]));o.s[i]=m/7;for(std::size_t j=0;j<c;j++){int q=int(std::lrint(w[i*c+j]/o.s[i]));q=std::max(-7,std::min(7,q));std::uint8_t code=q+8;auto&b=o.p[i*st+(j>>1)];if(j&1)b=(b&15)|(code<<4);else b=(b&240)|code;}}return o;}
static double us(std::function<void()> f,int n=100){for(int i=0;i<10;i++)f();auto a=std::chrono::steady_clock::now();for(int i=0;i<n;i++)f();auto b=std::chrono::steady_clock::now();return std::chrono::duration<double,std::micro>(b-a).count()/n;}
int main(){constexpr std::size_t M=1536,K=576,R=32;std::mt19937 g(9);std::normal_distribution<float>nd(0,1);std::vector<float>B(R*K),A(M*R),W(M*K),x(K),yd(M),yl(M),z(R);for(auto&a:B)a=nd(g)/std::sqrt(float(K));for(auto&a:A)a=nd(g)/std::sqrt(float(R));for(std::size_t i=0;i<M;i++)for(std::size_t j=0;j<K;j++){float s=0;for(std::size_t r=0;r<R;r++)s+=A[i*R+r]*B[r*K+j];W[i*K+j]=s+0.02f*nd(g);}for(auto&a:x)a=nd(g);auto qw=q4(W,M,K),qb=q4(B,R,K),qa=q4(A,M,R);double td=us([&]{larc::q4_gemv(qw.v(),x.data(),yd.data());});double tl=us([&]{larc::q4_projected_gemv(qb.v(),qa.v(),x.data(),z.data(),yl.data());});double se=0,sp=0;for(std::size_t i=0;i<M;i++){double e=double(yd[i])-yl[i];se+=e*e;sp+=double(yd[i])*yd[i];}auto bytes=[](const OQ&o){return o.p.size()+o.s.size()*sizeof(float);};auto db=bytes(qw),lb=bytes(qb)+bytes(qa);std::printf("dense_q4_bytes=%zu larc_factor_bytes=%zu resident_reduction=%.6f dense_us=%.6f larc_us=%.6f speedup=%.6f output_nmse=%.9f scratch_bytes=%zu\n",db,lb,double(db)/lb,td,tl,td/tl,se/sp,larc::projected_scratch_bytes(qb.v()));}
