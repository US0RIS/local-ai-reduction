#include "../runtime/larc_q2_attention.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
using namespace larc;

static std::uint8_t nearest_fp8(float x){float best=1e30f;int bi=0;for(int i=0;i<256;i++){float y=e4m3fn_to_float(std::uint8_t(i));if(!std::isfinite(y))continue;float d=std::fabs(x-y);if(d<best){best=d;bi=i;}}return std::uint8_t(bi);}
struct OwnedQ4{std::vector<std::uint8_t>p;std::vector<std::uint16_t>s;std::size_t r,c;Q4Rows view()const{return{p.data(),s.data(),r,c};}};
static OwnedQ4 q4(const std::vector<float>&w,std::size_t r,std::size_t c){OwnedQ4 o;o.r=r;o.c=c;o.s.resize(r);o.p.assign(r*((c+1)/2),0x88);auto st=(c+1)/2;for(std::size_t i=0;i<r;i++){float pos=0,neg=0;for(std::size_t j=0;j<c;j++){pos=std::max(pos,w[i*c+j]);neg=std::max(neg,-w[i*c+j]);}float sf=std::max(1e-8f,std::max(pos/7.0f,neg/8.0f));o.s[i]=float_to_fp16_bits(sf);float s=fp16_bits_to_float(o.s[i]);for(std::size_t j=0;j<c;j++){int q=int(std::lrint(w[i*c+j]/s));q=std::max(-8,std::min(7,q));std::uint8_t code=std::uint8_t(q+8);auto&b=o.p[i*st+(j>>1)];if(j&1)b=(b&15)|(code<<4);else b=(b&240)|code;}}return o;}
struct Cache{std::vector<std::uint8_t>p,mn,sc;std::size_t T,r;Q2RowsFP8 view()const{return{p.data(),mn.data(),sc.data(),T,r};}};
static Cache encode(const std::vector<float>&x,std::size_t T,std::size_t r){Cache c;c.T=T;c.r=r;auto st=(r+3)/4;c.p.assign(T*st,0);c.mn.resize(T);c.sc.resize(T);for(std::size_t t=0;t<T;t++){float mn=x[t*r],mx=mn;for(std::size_t j=1;j<r;j++){mn=std::min(mn,x[t*r+j]);mx=std::max(mx,x[t*r+j]);}float s=std::max((mx-mn)/3.0f,0.001953125f);c.mn[t]=nearest_fp8(mn);c.sc[t]=nearest_fp8(s);for(std::size_t j=0;j<r;j++){int q=int(std::lrint((x[t*r+j]-mn)/s));q=std::max(0,std::min(3,q));c.p[t*st+(j>>2)]|=std::uint8_t(q<<((j&3)*2));}}return c;}

int main(){
    // PyTorch/IEEE-style E4M3-FN golden values used by the Python codec.
    if(e4m3fn_to_float(0xBC)!=-1.5f || e4m3fn_to_float(0x38)!=1.0f || e4m3fn_to_float(0x7E)!=448.0f)return 2;
    constexpr std::size_t T=2048,R=16,D=32;std::mt19937 g(5);std::normal_distribution<float>n(0,0.2);std::vector<float>kb(R*D),vb(R*D),q(D),K(T*R),V(T*R);for(auto&z:kb)z=n(g);for(auto&z:vb)z=n(g);for(auto&z:q)z=n(g);for(auto&z:K)z=n(g);for(auto&z:V)z=n(g);
    auto kq=q4(kb,R,D);auto vq=q4(vb,R,D);auto ke=encode(K,T,R);auto ve=encode(V,T,R);std::vector<std::uint16_t>I(R*R,float_to_fp16_bits(0));for(std::size_t i=0;i<R;i++)I[i*R+i]=float_to_fp16_bits(1);
    std::vector<float>scratch(q2_attention_scratch_floats(T,R)),out(D),ref(D),ql(R),scores(T),vacc(R,0);latent_q2_fp8_attention_head(kq.view(),I.data(),vq.view(),I.data(),ke.view(),ve.view(),q.data(),1/std::sqrt(float(D)),scratch.data(),out.data());
    q4_gemv(kq.view(),q.data(),ql.data());float mx=-1e30f;auto st=(R+3)/4;for(std::size_t t=0;t<T;t++){float mn=e4m3fn_to_float(ke.mn[t]),sc=e4m3fn_to_float(ke.sc[t]),s=0;for(std::size_t j=0;j<R;j++){int code=(ke.p[t*st+(j>>2)]>>((j&3)*2))&3;s+=ql[j]*(mn+sc*code);}scores[t]=s/std::sqrt(float(D));mx=std::max(mx,scores[t]);}float den=0;for(auto&s:scores){s=std::exp(s-mx);den+=s;}for(std::size_t t=0;t<T;t++){float a=scores[t]/den,mn=e4m3fn_to_float(ve.mn[t]),sc=e4m3fn_to_float(ve.sc[t]);for(std::size_t j=0;j<R;j++){int code=(ve.p[t*st+(j>>2)]>>((j&3)*2))&3;vacc[j]+=a*(mn+sc*code);}}q4_transposed_gemv(vq.view(),vacc.data(),ref.data());
    float err=0;for(std::size_t i=0;i<D;i++)err=std::max(err,std::fabs(out[i]-ref[i]));auto cache_bytes=ke.p.size()+ke.mn.size()+ke.sc.size()+ve.p.size()+ve.mn.size()+ve.sc.size();std::printf("max_abs=%.9g scratch_bytes=%zu packed_cache_bytes=%zu decoded_latent_bytes=%zu\n",err,scratch.size()*sizeof(float),cache_bytes,2*T*R*sizeof(float));return err<1e-5f?0:1;
}
