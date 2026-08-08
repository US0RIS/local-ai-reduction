#include "../runtime/larc_q4.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

struct OwnedQ4 {
    std::vector<std::uint8_t> p; std::vector<float> s; std::size_t rows,cols;
    larc::Q4Rows view() const { return {p.data(),s.data(),rows,cols}; }
};

static OwnedQ4 q4(const std::vector<float>& w,std::size_t rows,std::size_t cols) {
    OwnedQ4 o; o.rows=rows;o.cols=cols;o.s.resize(rows);o.p.assign(rows*((cols+1)/2),0x88);
    const std::size_t stride=(cols+1)/2;
    for(std::size_t i=0;i<rows;i++) {
        float mx=1e-12f; for(std::size_t j=0;j<cols;j++) mx=std::max(mx,std::fabs(w[i*cols+j]));
        o.s[i]=mx/7.0f;
        for(std::size_t j=0;j<cols;j++) {
            int qv=int(std::lrint(w[i*cols+j]/o.s[i]));qv=std::max(-7,std::min(7,qv));std::uint8_t c=std::uint8_t(qv+8);
            auto& b=o.p[i*stride+(j>>1)]; if(j&1)b=(b&0x0F)|(c<<4);else b=(b&0xF0)|c;
        }
    }
    return o;
}
static std::vector<float> dq(const OwnedQ4&o) {
    std::vector<float>w(o.rows*o.cols);auto v=o.view();std::size_t st=(o.cols+1)/2;
    for(std::size_t i=0;i<o.rows;i++)for(std::size_t j=0;j<o.cols;j++){auto b=o.p[i*st+(j>>1)];int c=(j&1)?((b>>4)&15):(b&15);w[i*o.cols+j]=float(c-8)*o.s[i];}return w;
}
int main(){
    constexpr std::size_t K=97,R=19,M=83;std::mt19937 g(7);std::normal_distribution<float>n(0,1);
    std::vector<float>B(R*K),A(M*R),x(K);for(auto&z:B)z=n(g);for(auto&z:A)z=n(g);for(auto&z:x)z=n(g);
    auto qb=q4(B,R,K),qa=q4(A,M,R);std::vector<float>scratch(R),y(M),ref(M),z(R);larc::q4_projected_gemv(qb.view(),qa.view(),x.data(),scratch.data(),y.data());
    auto bd=dq(qb),ad=dq(qa);for(std::size_t i=0;i<R;i++){float a=0;for(std::size_t j=0;j<K;j++)a+=bd[i*K+j]*x[j];z[i]=a;}for(std::size_t i=0;i<M;i++){float a=0;for(std::size_t j=0;j<R;j++)a+=ad[i*R+j]*z[j];ref[i]=a;}
    float e=0;for(std::size_t i=0;i<M;i++)e=std::max(e,std::fabs(y[i]-ref[i]));std::printf("max_abs_error=%.9g scratch_bytes=%zu\n",e,larc::projected_scratch_bytes(qb.view()));return e<1e-3f?0:1;
}
