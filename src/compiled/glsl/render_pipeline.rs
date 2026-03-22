//! フルレンダリングパイプライン GLSL テンプレート
//!
//! ALICE-SDF-Experiment の描画技術を汎用テンプレート化。
//! `to_fragment_shader_full()` で SDF map 関数と結合してフルシェーダーを出力する。
//!
//! 含まれる技術:
//! - PBR (Cook-Torrance GGX BRDF, マルチスキャッタ補償, SSS)
//! - 大気散乱 (Rayleigh/Mie, オゾン吸収, 太陽/月/星/天の川/オーロラ)
//! - ボリュメトリック雲 (積雲/巻雲 2層)
//! - 天候 (霧, 雨, 雷)
//! - ポストプロセス (ACES トーンマップ, ブルーム, 色収差, ビネット, カラーグレーディング)
//! - レイマーチング最適化 (適応型ステップ, Analytic bloom 蓄積)
//!
//! Author: Moroya Sakamoto

/// レンダリング設定
#[derive(Debug, Clone)]
pub struct RenderConfig {
    /// レイマーチング最大ステップ数 (デフォルト: 128)
    pub max_steps: u32,
    /// レイマーチング最大距離 (デフォルト: 200.0)
    pub max_distance: f32,
    /// 昼夜サイクル有効 (デフォルト: true)
    pub day_night_cycle: bool,
    /// 天候システム有効 (デフォルト: true)
    pub weather_system: bool,
    /// SSR (Screen-Space Reflection) 有効 (デフォルト: true)
    pub ssr_enabled: bool,
    /// ボリュメトリックライト有効 (デフォルト: true)
    pub volumetric_light: bool,
    /// ポストプロセス有効 (デフォルト: true)
    pub post_process: bool,
    /// バイオーム地形有効 (デフォルト: false)
    pub biome_terrain: bool,
    /// GLSL バージョン (デフォルト: 300)
    pub glsl_version: u32,
    // ── Phase 1: メタバース還元 ──
    /// 分光レンダリング有効 — Planck黒体放射 + CIE 1931 + λ^-4散乱 (デフォルト: false)
    pub spectral_rendering: bool,
    /// マルチマテリアルスロット数 (デフォルト: 1)
    /// 1 = getDefaultMat()のみ、2+ = getMat(id, p) でID振り分け
    pub material_slots: u32,
    // ── Phase 2: 破壊 + VFX ──
    /// 物理破壊システム有効 — Voronoiクラック + 瓦礫 + 衝撃波 (デフォルト: false)
    pub destruction: bool,
    /// VFX: ドメインワープ + DBM放電 + 解析的ブルーム (デフォルト: false)
    pub vfx_effects: bool,
    // ── Phase 3: 仕上げ ──
    /// インテリアマッピング — ガラス壁の奥に疑似室内 (デフォルト: false)
    pub interior_mapping: bool,
    /// マイクロ法線ディテール — ナノスケール表面ディテール (デフォルト: false)
    pub micro_normal: bool,
    /// map_lite 軽量SDF二重構造 — AO/Shadow/Rain用 (デフォルト: false)
    pub dual_sdf: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            max_steps: 128,
            max_distance: 200.0,
            day_night_cycle: true,
            weather_system: true,
            ssr_enabled: true,
            volumetric_light: true,
            post_process: true,
            biome_terrain: false,
            glsl_version: 300,
            spectral_rendering: false,
            material_slots: 1,
            destruction: false,
            vfx_effects: false,
            interior_mapping: false,
            micro_normal: false,
            dual_sdf: false,
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GLSL テンプレートパーツ
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// ユニフォーム宣言（基本）
pub(crate) const UNIFORMS: &str = r"
uniform vec2 uRes;
uniform float uTime;
uniform vec3 uCamPos, uCamFwd, uCamRight, uCamUp;
uniform float uDayPhase;
uniform float uWxFog;
uniform float uWxRain;
uniform float uLightning;
uniform float uMaxDist;
";

/// 破壊システム用ユニフォーム
pub(crate) const DESTRUCTION_UNIFORMS: &str = r"
uniform float uEntropy;
uniform float uShatter;
uniform float uMeteorY;
uniform float uMeteorActive;
uniform vec2 uMeteorImpact;
uniform float uImpact;
uniform float uImpactRing;
uniform vec2 uShake;
uniform float uTimeDilation;
";

/// ノイズ関数群
pub(crate) const NOISE_LIB: &str = r"
// ═══ Noise ═══
#define SAT(x) clamp(x,0.0,1.0)
float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
float hash3(vec3 p){return fract(sin(dot(p,vec3(127.1,311.7,74.7)))*43758.5453);}
float vnoise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.0-2.0*f);return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x),f.y);}
float vnoise3(vec3 p){vec3 i=floor(p),f=fract(p);f=f*f*(3.0-2.0*f);float n00=mix(hash3(i),hash3(i+vec3(1,0,0)),f.x);float n10=mix(hash3(i+vec3(0,1,0)),hash3(i+vec3(1,1,0)),f.x);float n01=mix(hash3(i+vec3(0,0,1)),hash3(i+vec3(1,0,1)),f.x);float n11=mix(hash3(i+vec3(0,1,1)),hash3(i+vec3(1,1,1)),f.x);return mix(mix(n00,n10,f.y),mix(n01,n11,f.y),f.z);}
float fbm(vec2 p){float v=0.0,a=0.5;mat2 r=mat2(0.8,0.6,-0.6,0.8);for(int i=0;i<3;i++){v+=a*vnoise(p);p=r*p*2.1;a*=0.48;}return v;}
float fbm3(vec3 p){float v=0.0,a=0.5;for(int i=0;i<3;i++){v+=a*vnoise3(p);p=p*2.15+vec3(1.7,3.2,2.8);a*=0.45;}return v;}
";

/// バイオームシステム（真理の地形法）
pub(crate) const BIOME_SYSTEM: &str = r"
// ═══ Biome System (真理の地形法 — ズートピア型ラジアル配置) ═══
#define PI 3.14159265
#define TAU 6.28318530
float angleDist(float a,float b){float d=a-b;d=d-TAU*floor((d+PI)/TAU);return abs(d);}
vec4 biomeWeights(vec2 xz){
  float dist=length(xz);
  float ang=atan(xz.y,xz.x);
  float aSnow=PI*0.25,aRock=-PI*0.25,aDesert=-PI*0.75,aGrass=PI*0.75;
  float wS=exp(-angleDist(ang,aSnow)*angleDist(ang,aSnow)*3.0);
  float wR=exp(-angleDist(ang,aRock)*angleDist(ang,aRock)*3.0);
  float wD=exp(-angleDist(ang,aDesert)*angleDist(ang,aDesert)*3.0);
  float wG=exp(-angleDist(ang,aGrass)*angleDist(ang,aGrass)*3.0);
  float hub=smoothstep(12.0,5.0,dist);
  wS=mix(wS,1.0,hub);wR=mix(wR,1.0,hub);wD=mix(wD,1.0,hub);wG=mix(wG,1.0,hub);
  float invSum=1.0/(wS+wR+wD+wG+0.001);
  return vec4(wS,wD,wR,wG)*invSum;
}
float voronoiErosion(vec2 p){
  vec2 n=floor(p);vec2 f=fract(p);float md=8.0,md2=8.0;
  for(int j=-1;j<=1;j++)for(int i=-1;i<=1;i++){
    vec2 g=vec2(float(i),float(j));vec2 o=vec2(hash(n+g),hash(n+g+vec2(31.3,17.7)));
    vec2 r=g+o-f;float d=dot(r,r);float sel=step(d,md);md2=mix(md2,md,sel);md=mix(md,d,sel);
  }
  return sqrt(md)*0.6-(sqrt(md2)-sqrt(md))*0.8;
}
float terrainHeight(vec2 xz){
  vec4 w=biomeWeights(xz);
  float hSnow=fbm(xz*0.12)*1.5+vnoise(xz*0.3)*0.4;
  vec2 windDir=vec2(0.8,0.6);float windProj=dot(xz*0.08,windDir);float windPerp=dot(xz*0.08,vec2(-windDir.y,windDir.x));
  float hDesert=fbm(vec2(windProj*3.0,windPerp*0.8))*0.8+vnoise(xz*0.06)*0.5;
  float hRock=voronoiErosion(xz*0.15)*2.2;
  float hGrass=fbm(xz*0.1)*1.0+vnoise(xz*0.2)*0.25;
  return w.x*hSnow+w.y*hDesert+w.z*hRock+w.w*hGrass;
}
";

/// 分光レンダリング (Planck黒体放射 + CIE 1931 + λ^-4散乱)
pub(crate) const SPECTRAL_LIB: &str = r"
// ═══ Spectral Rendering (ALICE-VFX Recipe) ═══
vec3 blackbody(float K){
  float t=K*0.01;
  float hi=step(66.0,t);float lo=1.0-hi;
  float r=mix(1.0,SAT(1.292936*inversesqrt(max(t-55.0,0.001))-0.16),hi);
  float g=mix(SAT(0.39008*log(max(t,1.0))-0.63184),SAT(1.129891*inversesqrt(max(t-50.0,0.001))-0.15),hi);
  float warm=step(19.0,t);
  float b=mix(SAT(0.54321*log(max(t-10.0,1.0))-1.19625)*warm,1.0,hi);
  return vec3(r,g,b);
}
vec3 spectralToRGB(float lambda){
  float x=1.056*exp(-0.5*pow((lambda-599.8)*0.0244,2.0))+0.362*exp(-0.5*pow((lambda-442.0)*0.0624,2.0))-0.065*exp(-0.5*pow((lambda-501.1)*0.049,2.0));
  float y=0.821*exp(-0.5*pow((lambda-568.8)*0.0213,2.0))+0.286*exp(-0.5*pow((lambda-530.9)*0.0613,2.0));
  float z=1.217*exp(-0.5*pow((lambda-437.0)*0.0845,2.0))+0.681*exp(-0.5*pow((lambda-459.0)*0.0385,2.0));
  return vec3(3.2406*x-1.5372*y-0.4986*z,-0.9689*x+1.8758*y+0.0415*z,0.0557*x-0.2040*y+1.0570*z);
}
vec3 spectralBlackbody(float K){
  vec3 acc=vec3(0);float invK=1.0/(K+0.001);
  for(int i=0;i<4;i++){
    float lambda=380.0+float(i)*113.3;
    float lm=lambda*1e-3;
    float x=1.4388e3*invK/(lambda+0.001);
    float planck=1.0/(lm*lm*lm*lm*lm*(exp(min(x,80.0))-1.0)+0.001);
    acc+=spectralToRGB(lambda)*planck;
  }
  return max(acc*0.00025,vec3(0));
}
vec3 rayleighSpectral(float mu,float am,float densR,vec3 extR){
  vec3 acc=vec3(0);
  for(int i=0;i<4;i++){
    float lambda=400.0+float(i)*100.0;
    float scatter=1.0/(lambda*lambda*lambda*lambda)*1e10;
    float phR=0.059683*(1.0+mu*mu);
    vec3 rgb=spectralToRGB(lambda);
    acc+=rgb*scatter*phR*am*densR;
  }
  return max(acc*extR*0.012,vec3(0));
}
";

/// VFXエフェクト (ドメインワープ + DBM放電 + 解析的ブルーム)
pub(crate) const VFX_LIB: &str = r"
// ═══ VFX Effects ═══
vec3 aBloom(float d,vec3 gc,float intensity,float falloff){
  return gc*exp(-abs(d)*falloff)*intensity;
}
vec3 dWarp(vec3 p,float t,float intensity){
  float fx=sin(p.y*1.7+t*0.3);float fy=cos(p.z*1.3+t*0.5);float fz=sin(p.x*2.1+t*0.4);
  return p+vec3(fx,fy,fz)*intensity;
}
float dbmDischarge(vec3 p,vec3 src,float charge,float t){
  vec3 dir=normalize(p-src);float dist=length(p-src);
  float discharge=0.0;float amp=1.0;vec3 q=p;
  for(int i=0;i<6;i++){
    q=abs(q)-dir*0.5*amp;
    float ca=cos(charge*float(i)+t);float sa=sin(charge*float(i)+t);
    q.xy=vec2(q.x*ca-q.y*sa,q.x*sa+q.y*ca);
    discharge+=(1.0/(length(q.xz)+0.01))*amp;
    amp*=0.5;
  }
  return discharge*exp(-dist*0.5)*charge;
}
";

/// マイクロ法線ディテール (ナノスケール表面)
pub(crate) const MICRO_NORMAL_LIB: &str = r"
// ═══ Micro Normal (nanoscale thermal vibration) ═══
vec3 microNormal(vec3 p,vec3 n,float freq,float amp){
  vec2 e=vec2(0.001,0);
  float d0=vnoise3(p*freq);
  float dx=vnoise3((p+e.xyy)*freq)-d0;
  float dy=vnoise3((p+e.yxy)*freq)-d0;
  float dz=vnoise3((p+e.yyx)*freq)-d0;
  return normalize(n+vec3(dx,dy,dz)*(amp/(e.x+0.0001)));
}
";

/// インテリアマッピング (ガラス壁の奥に疑似室内)
pub(crate) const INTERIOR_MAPPING_LIB: &str = r"
// ═══ Interior Mapping (pseudo-rooms behind walls) ═══
vec3 interiorMap(vec3 p,float scale){
  vec3 uv=fract(p*scale);
  vec3 fp=abs(uv-0.5);
  float wall=smoothstep(0.02,0.0,min(fp.x,fp.z));
  float flr=smoothstep(0.02,0.0,fp.y);
  float room=hash(floor(p.xz*scale));
  float lit=step(0.6,room);
  vec3 col=mix(vec3(0.005,0.01,0.025),vec3(0.02,0.06,0.15),lit);
  col+=vec3(0.08,0.25,0.65)*wall*lit*0.3;
  col+=vec3(0.04,0.12,0.3)*flr*0.15;
  return col;
}
";

/// 物理破壊システム (Voronoiクラック + 瓦礫 + 衝撃波)
pub(crate) const DESTRUCTION_SYSTEM: &str = r"
// ═══ Physical Destruction (Law D-2) ═══
vec3 voronoi2(vec2 p){
  vec2 n=floor(p);vec2 f=fract(p);
  float md=8.0,md2=8.0;vec2 mg=vec2(0);
  for(int j=-1;j<=1;j++)for(int i=-1;i<=1;i++){
    vec2 g=vec2(float(i),float(j));
    vec2 o=vec2(hash(n+g),hash(n+g+vec2(31.3,17.7)));
    vec2 r=g+o-f;float d=dot(r,r);
    float sel=step(d,md);md2=mix(md2,md,sel);md=mix(md,d,sel);mg=mix(mg,n+g,sel);
  }
  return vec3(sqrt(md),sqrt(md2)-sqrt(md),hash(mg));
}
float destructionAt(vec3 p,float impactRing,float shatter,vec2 impactXZ){
  float waveR=impactRing*0.5;
  float cd=length(p.xz-impactXZ);
  float waveDist=abs(cd-waveR);
  float waveW=max(waveR*0.35,2.0);
  float invW=1.0/(waveW+0.001);
  float wave=SAT(1.0-waveDist*invW);
  float coreR=max(impactRing*0.12,1.0);
  float core=SAT(1.0-cd*(1.0/(coreR+0.001)));
  return max(wave,core)*shatter;
}
float sdDestruction(vec3 p,float orig,float destr){
  float gate=step(0.005,destr);
  vec3 v=voronoi2(p.xz*5.0);
  float crack=v.y;
  crack=smoothstep(0.0,0.05*(1.0-destr*0.95),crack);
  float cracked=orig+(1.0-crack)*0.1*destr;
  float cellHash=v.z;
  float remove=step(cellHash,destr*0.7)*gate;
  cracked=mix(cracked,1e5,remove);
  return mix(orig,cracked,gate);
}
float sdDebris(vec3 p,float t,vec2 center){
  float gate=step(0.15,t);
  vec2 cellId=floor(p.xz*3.0);
  float h=hash(cellId+center*7.13);
  float fallGate=step(h,t*1.2)*gate;
  float fallTime=max(t-h,0.0);
  vec3 q=p;
  q.y+=4.9*fallTime*fallTime;
  q.xz+=(vec2(hash(cellId+vec2(5.3,1.7)),hash(cellId+vec2(9.1,3.2)))-0.5)*fallTime*0.5;
  float ca=cos(fallTime*(h*5.0+1.0));float sa=sin(fallTime*(h*5.0+1.0));
  q.xy=vec2(q.x*ca-q.y*sa,q.x*sa+q.y*ca);
  float sz=0.08+h*0.12;
  vec3 bq=abs(q-vec3(0,sz,0))-vec3(sz,sz*0.5,sz*0.7);
  float db=length(max(bq,0.0))+min(max(bq.x,max(bq.y,bq.z)),0.0);
  return mix(1e5,db,fallGate);
}
";

/// PBR マテリアル構造体 + Cook-Torrance BRDF
pub(crate) const PBR_BRDF: &str = r"
// ═══ PBR (Cook-Torrance GGX) ═══
struct Mat{vec3 albedo;float metallic;float roughness;vec3 emission;float sss;};

Mat getDefaultMat(vec3 p){
  Mat m;m.albedo=vec3(0.5);m.metallic=0.0;m.roughness=0.5;m.emission=vec3(0);m.sss=0.0;
  return m;
}

vec3 pbrDirect(vec3 n,vec3 v,vec3 l,vec3 lc,Mat m,vec3 F0){
  vec3 h=normalize(v+l);
  float NdL=max(dot(n,l),0.0);float NdH=max(dot(n,h),0.0);float NdV=max(dot(n,v),0.001);float VdH=max(dot(v,h),0.001);
  float r2=m.roughness*m.roughness;float r4=r2*r2;
  // GGX NDF
  float denom=NdH*NdH*(r4-1.0)+1.0;float D=r4/(3.14159265*denom*denom+0.0001);
  // Schlick-GGX geometry
  float k=r2*0.5;float G1v=NdV/(NdV*(1.0-k)+k);float G1l=NdL/(NdL*(1.0-k)+k);float G=G1v*G1l;
  // Fresnel (Schlick)
  float ft=1.0-VdH;float ft2=ft*ft;vec3 F=F0+(1.0-F0)*(ft2*ft2*ft);
  vec3 spec=D*G*F/(4.0*NdV*NdL+0.0001);
  vec3 kD=(1.0-F)*(1.0-m.metallic);
  return (kD*m.albedo/3.14159265+spec)*lc*NdL;
}

vec3 pointLight(vec3 p,vec3 n,vec3 v,Mat m,vec3 F0,vec3 lp,vec3 lc,float lr){
  vec3 L=lp-p;float dist2=dot(L,L);L=normalize(L);
  float atten=lr/(dist2+1.0);
  return pbrDirect(n,v,L,lc*atten,m,F0);
}
";

/// 法線計算 + AO + シャドウ (AO/Shadowはmap_liteを使用)
pub(crate) const NORMAL_AO_SHADOW: &str = r"
// ═══ Normal / AO / Shadow ═══
vec3 calcN(vec3 p,float t){
  vec2 e=vec2(max(0.0002,t*0.0003),0);
  return normalize(vec3(
    sdf_map(p+e.xyy).x-sdf_map(p-e.xyy).x,
    sdf_map(p+e.yxy).x-sdf_map(p-e.yxy).x,
    sdf_map(p+e.yyx).x-sdf_map(p-e.yyx).x));
}
float ao(vec3 p,vec3 n){
  float occ=0.0;float sc=1.0;
  for(int i=1;i<=6;i++){float hr=0.008+0.12*float(i);float dd=sdf_map_lite(p+n*hr);occ+=(hr-dd)*sc;sc*=0.58;}
  return SAT(1.0-4.0*occ);
}
float shadowProxy(vec3 p,vec3 n,vec3 l){
  float h1=0.1,h2=0.4;float d1=sdf_map_lite(p+l*h1);float d2=sdf_map_lite(p+l*h2);
  float occ=SAT((d1/h1+d2/h2)*0.5);float NdL=max(dot(n,l),0.0);return occ*NdL;
}
float rainOcc(vec3 p){
  float t=0.3;
  for(int i=0;i<3;i++){float h=sdf_map_lite(p+vec3(0,t,0));if(h<0.08)return 0.0;t+=max(h,0.5);}
  return 1.0;
}
";

/// 大気散乱（空、太陽、月、星、雲）
pub(crate) const SKY_ATMOSPHERE: &str = r"
// ═══ Sky (Rayleigh + Mie + Stars + Clouds) ═══
vec3 skyColor(vec3 rd,vec3 sunDir,vec3 moonDir,float dayF){
  float y=rd.y;float mu=dot(rd,sunDir);float sunH=sunDir.y;
  float goldenF=exp(-sunH*sunH*12.0)*smoothstep(-0.15,0.05,sunH);
  float nightF=1.0-dayF;
  // Scattering coefficients
  vec3 bR=vec3(5.8,13.5,33.1)*1e-3;float bM=3.0e-3;
  float am=1.0/(abs(y)+0.15);
  float sunCZ=max(sunH,0.0)+0.15;float sunAm=1.0/(sunCZ+0.15*pow(sunCZ,0.6));
  float densR=exp(-max(y,0.0)*3.0);float densM=exp(-max(y,0.0)*1.2);
  vec3 extR=exp(-bR*sunAm*1.5);float extM=exp(-bM*sunAm*0.8);
  // Rayleigh
  float phR=0.059683*(1.0+mu*mu);vec3 rayleigh=bR*phR*am*densR*extR;
  // Mie
  float g=0.76,g2=g*g;float denomMie=max(1.0+g2-2.0*g*mu,0.0001);
  float denomSqrt=inversesqrt(denomMie);float phM=0.079577*(1.0-g2)*denomSqrt*denomSqrt*denomSqrt;
  vec3 mie=vec3(bM*phM*(am*0.5)*densM)*extR*extM;
  // Ozone
  vec3 bO=vec3(0.065,0.19,0.005)*1e-2;float ozoneAm=1.0/(max(sunH+0.1,0.001)+0.05);
  float ozoneF=smoothstep(-0.1,-0.02,sunH)*smoothstep(0.15,0.04,sunH);vec3 ozoneExt=exp(-bO*ozoneAm*5.0)*ozoneF;
  // Combined
  vec3 sunI=vec3(22.0,20.0,17.0)*smoothstep(-0.08,0.15,sunH);
  sunI*=mix(vec3(1),ozoneExt+vec3(0.3,0.2,0.8),ozoneF);
  vec3 sky=(rayleigh+mie)*sunI;
  sky*=mix(vec3(1),vec3(0.7,0.75,1.3),ozoneF*0.4);
  sky+=vec3(0.001,0.004,0.025)*ozoneF*smoothstep(-0.1,0.3,y);
  sky+=vec3(0.0,0.003,0.015)*max(y,0.0)*dayF;
  // Sun disc (limb darkening)
  float sunAng=acos(clamp(mu,-1.0,1.0));float sunR=0.0046;
  float sunDisc=smoothstep(sunR*1.3,sunR*0.4,sunAng);
  float limbT=min(sunAng/sunR,1.0);float limb=1.0-0.6*(1.0-sqrt(max(1.0-limbT*limbT,0.0)));
  sky+=vec3(12.0,10.0,7.0)*sunDisc*max(limb,0.0)*smoothstep(-0.05,0.05,sunH);
  // Golden hour
  sky+=vec3(0.35,0.12,0.03)*goldenF*exp(-abs(y)*3.0)*0.5;
  sky+=vec3(0.6,0.25,0.08)*goldenF*pow(max(mu,0.0),4.0)*0.2;
  // Moon
  float moonAng=acos(clamp(dot(rd,moonDir),-1.0,1.0));
  sky+=vec3(0.5,0.55,0.65)*smoothstep(0.009,0.003,moonAng)*nightF*1.5;
  // Stars
  vec3 sid=floor(rd*420.0);float ss=hash3(sid);float mag=pow(ss,0.25);
  float starB=smoothstep(0.88,1.0,mag)*0.5*(0.5+0.5*sin(uTime*(hash3(sid+200.0)*3.5+0.5)));
  float bv=hash3(sid+300.0);
  vec3 starC=mix(mix(vec3(0.6,0.7,1.0),vec3(1.0,0.97,0.93),smoothstep(0.0,0.4,bv)),mix(vec3(1.0,0.85,0.65),vec3(1.0,0.6,0.35),smoothstep(0.5,1.0,bv)),smoothstep(0.35,0.55,bv));
  sky+=starC*starB*nightF*smoothstep(0.0,0.08,y);
  // Milky Way
  float mwAng=acos(clamp(abs(dot(rd,normalize(vec3(0.3,0.7,0.15)))),0.0,1.0));
  sky+=vec3(0.045,0.03,0.06)*exp(-(mwAng-0.2)*(mwAng-0.2)*6.0)*vnoise(rd.xz*3.5+rd.y*2.0)*nightF*smoothstep(0.05,0.35,y);
  // Clouds (dual layer)
  if(y>0.008){
    float invY=1.0/y;
    vec2 cUV=rd.xz*invY*0.12+uTime*vec2(0.003,0.001);float cn=fbm(cUV*4.0);float cn2=vnoise(cUV*16.0+30.0);
    float cover=0.08+uWxFog*0.35+uWxRain*0.4;
    float cD=smoothstep(0.4-cover,0.7,cn+cn2*0.15)*smoothstep(0.008,0.12,y);
    float cLit=smoothstep(0.35,0.75,cn)*0.6+0.4;cLit*=max(sunH+0.2,0.08);
    vec3 cBr=mix(vec3(0.04,0.045,0.06),vec3(1.0,0.95,0.85),dayF)*cLit;
    vec3 cDk=mix(vec3(0.012,0.015,0.025),vec3(0.3,0.3,0.35),dayF);
    cBr+=vec3(1.0,0.45,0.15)*goldenF*0.8;cDk+=vec3(0.5,0.2,0.08)*goldenF*0.3;
    sky=mix(sky,mix(cDk,cBr,cLit),SAT(cD));
    // Cirrus
    vec2 ciUV=rd.xz*invY*0.05+uTime*vec2(0.005,0.002);float ci=fbm(ciUV*10.0);
    float ciD=smoothstep(0.52,0.78,ci)*0.3*smoothstep(0.1,0.35,y);
    sky=mix(sky,mix(vec3(0.025,0.03,0.04),vec3(0.55,0.55,0.6),dayF)+vec3(0.4,0.2,0.08)*goldenF*0.4,SAT(ciD));
  }
  sky=mix(sky,mix(vec3(0.02,0.025,0.04),vec3(0.3,0.33,0.38),dayF),uWxFog*0.55);
  return max(sky,vec3(0));
}
";

/// ボリュメトリックライトスキャッタ
pub(crate) const VOLUMETRIC_LIGHT: &str = r"
// ═══ Volumetric Light Scatter ═══
vec3 volScatter(vec3 ro,vec3 rd,vec3 sunDir,float maxDist,float dayF,vec2 fragCoord){
  float dither=fract(dot(fragCoord,vec2(0.7548776662,0.5698402909)));
  float stepSize=maxDist/8.0;vec3 acc=vec3(0);
  for(int i=0;i<8;i++){
    float t=stepSize*(float(i)+dither);vec3 sp=ro+rd*t;
    float sd=sdf_map(sp).x;float vis=smoothstep(0.0,0.5,sd);
    float scatter=vis*exp(-t*0.04);
    float mu=dot(rd,sunDir);float phase=0.079577*(1.0+mu*mu);
    acc+=phase*scatter*stepSize;
  }
  vec3 lc=mix(vec3(0.01,0.015,0.04),vec3(0.8,0.7,0.5),dayF);
  return acc*lc*0.15;
}
";

/// ポストプロセス（ACES, ブルーム, CA, ビネット, カラーグレーディング）
pub(crate) const POST_PROCESS: &str = r"
// ═══ Post Process ═══
vec3 postProcess(vec3 col,vec2 fragCoord,vec2 res){
  // ACES fitted
  col*=0.65;col=SAT((col*(2.51*col+0.03))/(col*(2.43*col+0.59)+0.14));
  // HDR Bloom
  float lum=dot(col,vec3(0.2126,0.7152,0.0722));
  col+=col*max(lum-0.38,0.0)*0.85;
  // Chromatic aberration
  vec2 caUV=fragCoord/res-0.5;float caDist=dot(caUV,caUV);
  float caStr=caDist*0.012+max(lum-0.7,0.0)*0.06;
  col.r*=1.0+caStr;col.b*=1.0-caStr*0.8;
  // Vignette
  float vig=1.0-caDist*1.8*0.45;vig*=vig;col*=vig;
  // Film grain
  col+=(hash(fragCoord)-0.5)*0.018*(1.0-lum*0.6);
  // Color grading (3-way split)
  float lumF=dot(col,vec3(0.2126,0.7152,0.0722));
  float sW=1.0-smoothstep(0.0,0.35,lumF);float hW=smoothstep(0.55,1.0,lumF);float mW=1.0-sW-hW;
  col+=vec3(-0.006,0.008,0.022)*sW*0.5+vec3(0.002,0.004,0.006)*mW*0.3+vec3(0.018,0.008,-0.01)*hW*0.4;
  // Blue noise dither
  col+=(fract(dot(fragCoord,vec2(0.7548776662,0.5698402909)))-0.5)*0.00392157;
  // Gamma
  col=pow(max(col,vec3(0)),vec3(1.0/2.2));
  return col;
}
";

/// メインレンダリングループ（レイマーチ + ライティング + 大気 + ポスト）を組み立て
pub(crate) fn build_main_function(config: &RenderConfig) -> String {
    let weather_uniforms = if config.weather_system {
        ""
    } else {
        "\n#define uWxFog 0.0\n#define uWxRain 0.0\n#define uLightning 0.0\n"
    };
    let destruction_defines = if !config.destruction {
        "\n#define uEntropy 0.0\n#define uShatter 0.0\n#define uMeteorY 300.0\n#define uMeteorActive 0.0\n#define uImpact 0.0\n#define uImpactRing 0.0\n"
    } else {
        ""
    };

    let day_night = if config.day_night_cycle {
        r"
  float sunAngle=uDayPhase*TAU;
  float sunH=sin(sunAngle);
  vec3 sunDir=normalize(vec3(cos(sunAngle)*0.8,sunH,0.35));
  vec3 moonDir=normalize(vec3(-cos(sunAngle)*0.6,max(-sunH*0.8,0.15),-0.3));
  float dayF=smoothstep(-0.1,0.3,sunH);
  float goldenF=exp(-sunH*sunH*12.0)*smoothstep(-0.15,0.05,sunH);"
    } else {
        r"
  vec3 sunDir=normalize(vec3(0.5,0.7,0.35));
  vec3 moonDir=normalize(vec3(-0.5,0.3,-0.3));
  float dayF=1.0;float goldenF=0.0;float sunH=0.7;"
    };

    let camera_shake = if config.destruction {
        r"
  vec2 sUV=uv+uShake;
  if(uImpact>0.1){sUV+=vec2(sin(uv.y*35.0+uTime*8.0),cos(uv.x*28.0+uTime*6.0))*uImpact*0.012;}
  vec3 rd=normalize(uCamFwd+sUV.x*uCamRight+sUV.y*uCamUp);"
    } else {
        "  vec3 rd=normalize(uCamFwd+uv.x*uCamRight+uv.y*uCamUp);"
    };

    let bloom_accum = if config.vfx_effects {
        r"
  vec3 bloomAcc=vec3(0);"
    } else {
        ""
    };

    let bloom_in_loop = if config.vfx_effects {
        r"
    float bm1=step(0.5,hit.y)*step(hit.y,1.5);
    float bm2=step(16.5,hit.y)*step(hit.y,17.5);
    bloomAcc+=bm1*aBloom(hit.x,vec3(0.3,0.5,1.0),0.012,8.0)+bm2*aBloom(hit.x,vec3(0.4,0.6,1.2),0.008,10.0);"
    } else {
        ""
    };

    let bloom_to_col = if config.vfx_effects {
        "  col+=bloomAcc;"
    } else {
        ""
    };

    let get_mat = if config.material_slots > 1 {
        "    Mat mat=getMat(hit.y,p);"
    } else {
        "    Mat mat=getDefaultMat(p);\n    mat.emission=vec3(0);"
    };

    let rain_occ = if config.weather_system {
        r"
    float rocc=1.0;if(uWxRain>0.01)rocc=rainOcc(p);
    if(hit.y<0.5)mat.roughness=mix(mat.roughness,0.18,1.0-rocc);"
    } else {
        "    float rocc=1.0;"
    };

    let micro_normal_apply = if config.micro_normal {
        r"
    n=microNormal(p,n,80.0,0.3);"
    } else {
        ""
    };

    let vol_light = if config.volumetric_light {
        r"
  // Volumetric light
  float volDist=min(t,40.0);
  vec3 vol=volScatter(ro,rd,sunDir,volDist,dayF,gl_FragCoord.xy);
  col+=vol*(1.0-totalFog);
  float godRayS=max(dot(rd,sunDir),0.0);
  vec3 godCol=mix(vec3(0.5,0.3,0.15),vec3(1.0,0.95,0.8),dayF);
  godCol=mix(godCol,vec3(0.8,0.35,0.1),goldenF*0.5);
  col+=godCol*(pow(godRayS,48.0)*0.1+pow(godRayS,8.0)*0.02)*dayF*(1.0-totalFog);
  float moonGR=pow(max(dot(rd,moonDir),0.0),32.0)*0.03*(1.0-dayF);
  col+=vec3(0.08,0.1,0.15)*moonGR*(1.0-totalFog);"
    } else {
        ""
    };

    let destruction_atmosphere = if config.destruction {
        r"
  // Meteor atmosphere
  if(uMeteorActive>0.5){
    vec3 mwp=vec3(uMeteorImpact.x,max(uMeteorY,0.5),uMeteorImpact.y);
    vec3 toM=normalize(mwp-ro);float mDot=max(dot(rd,toM),0.0);
    col+=vec3(8.0,4.0,1.5)*pow(mDot,32.0)*0.6;
    col+=vec3(1.5,0.4,0.08)*pow(mDot,6.0)*0.4;
    float skyRed=SAT(1.0-uMeteorY*0.005)*0.45;
    col=mix(col,col*vec3(1.5,0.55,0.35),skyRed);
  }
  // Entropy atmosphere tint
  if(uEntropy>0.3){
    float eFade=(uEntropy-0.3)*1.43;
    col=mix(col,col*vec3(1.15,0.85,0.7),eFade*0.12);
  }
  // Impact flash
  col+=vec3(5.0,3.5,2.0)*max(uImpact-0.6,0.0)*3.0;"
    } else {
        ""
    };

    let shatter_cracks = if config.destruction {
        r"
    // Glass shatter cracks
    if(uShatter>0.005){
      vec3 sp=p*0.5;vec3 cell=floor(sp);vec3 fr=fract(sp);
      vec3 df=min(fr,1.0-fr);float edist=min(df.x,min(df.y,df.z));
      float crack=smoothstep(0.12*uShatter,0.0,edist);
      vec3 cCol=mix(vec3(1.2,0.4,0.06),vec3(0.15,0.4,1.0),1.0-uEntropy);
      col+=cCol*crack*uShatter*2.5;
      col*=1.0-crack*0.55*uShatter;
    }
    // Impact shockwave ring
    if(uImpact>0.01&&uImpactRing>1.0){
      float rDist=abs(length(p.xz-uMeteorImpact)-uImpactRing);
      float ring=exp(-rDist*1.5)*uImpact;
      col+=mix(vec3(4.0,2.8,1.2),vec3(1.5,0.5,0.1),smoothstep(0.0,3.0,rDist))*ring*0.7;
    }"
    } else {
        ""
    };

    let ssr = if config.ssr_enabled {
        r"
    // Floor SSR
    if(hit.y<0.5){
      vec3 reflDir=reflect(rd,n);
      float rt=0.0;vec2 rh;
      for(int i=0;i<16;i++){rh=sdf_map(p+reflDir*rt);if(rh.x<0.001||rt>45.0)break;rt+=rh.x;}
      vec3 reflCol=skyColor(reflDir,sunDir,moonDir,dayF);
      if(rt<45.0){
        vec3 rp=p+reflDir*rt;vec3 rn=calcN(rp,t+rt);
        float rNdL=max(dot(rn,keyDir),0.0);
        reflCol*=rNdL*0.5;reflCol*=exp(-rt*0.035);
      }
      float reflStr=envBRDF.x+envBRDF.y;
      float rainWet=uWxRain*0.35*rocc;
      reflStr=max(reflStr,rainWet);
      col=mix(col,reflCol,reflStr*(1.0-mat.roughness*0.8));
    }"
    } else {
        ""
    };

    let post = if config.post_process {
        "  col=postProcess(col,gl_FragCoord.xy,uRes);"
    } else {
        "  col=pow(max(col,vec3(0)),vec3(1.0/2.2));"
    };

    let rain_streaks = if config.weather_system {
        r"
  // Rain streaks
  if(uWxRain>0.01){
    float rainDepth=smoothstep(3.0,35.0,t);rainDepth*=smoothstep(-0.5,-0.1,rd.y);
    vec2 ruv=gl_FragCoord.xy/uRes;
    vec2 rq=ruv*vec2(25.0,7.0);rq.y+=uTime*4.0;rq.x+=uTime*0.3;
    float r1=smoothstep(0.04,0.0,abs(fract(rq).x-0.5))*fract(rq).y*step(0.9,hash(floor(rq)));
    rq=ruv*vec2(50.0,12.0);rq.y+=uTime*5.5;rq.x-=uTime*0.15;
    float r2=smoothstep(0.025,0.0,abs(fract(rq).x-0.5))*fract(rq).y*step(0.92,hash(floor(rq)+300.0));
    col+=vec3(0.25,0.3,0.4)*(r1*0.3+r2*0.15)*uWxRain*rainDepth;
  }
  col+=vec3(0.5,0.55,0.7)*uLightning*0.6;"
    } else {
        ""
    };

    format!(
        r"
{weather_uniforms}
{destruction_defines}
void main(){{
  vec2 uv=(gl_FragCoord.xy-0.5*uRes)/uRes.y;
  vec3 ro=uCamPos;
{camera_shake}
{day_night}
  vec3 sky=skyColor(rd,sunDir,moonDir,dayF);
{bloom_accum}

  // Raymarch
  float t=0.0;vec2 hit;
  for(int i=0;i<{max_steps};i++){{
    hit=sdf_map(ro+rd*t);
{bloom_in_loop}
    if(hit.x<0.0004||t>{max_dist})break;
    t+=hit.x*max(1.0,t*0.02);
  }}

  vec3 col=sky;
{bloom_to_col}

  if(t<{max_dist}){{
    vec3 p=ro+rd*t;
    vec3 n=calcN(p,t);
    vec3 V=-rd;
{get_mat}
{rain_occ}
{micro_normal}
    vec3 F0=mix(vec3(0.04),mat.albedo,mat.metallic);

    // Lighting
    vec3 keyDir=mix(moonDir,sunDir,dayF);
    vec3 keyCol=mix(vec3(0.06,0.08,0.18),vec3(1.5,1.35,1.1),dayF);
    keyCol=mix(keyCol,vec3(1.5,0.6,0.25),goldenF*0.7);
    keyCol+=vec3(2.0,2.2,2.8)*uLightning;
    float keyShadow=shadowProxy(p+n*0.015,n,keyDir);
    vec3 fillDir=normalize(vec3(-0.35,0.35,-0.6));
    vec3 fillCol=mix(vec3(0.03,0.04,0.08),vec3(0.18,0.25,0.45),dayF);
    vec3 rimDir=normalize(vec3(0.0,0.25,-0.9));
    vec3 rimCol=mix(vec3(0.04,0.05,0.1),vec3(0.25,0.35,0.5),dayF);
    float occ=ao(p,n);

    vec3 Lo=pbrDirect(n,V,keyDir,keyCol,mat,F0)*keyShadow;
    Lo+=pbrDirect(n,V,fillDir,fillCol,mat,F0);
    Lo+=pbrDirect(n,V,rimDir,rimCol,mat,F0);

    // IBL ambient (6-direction irradiance probe)
    vec3 skyUp=mix(vec3(0.008,0.012,0.03),vec3(0.1,0.14,0.25),dayF);
    vec3 skyDn=mix(vec3(0.003,0.005,0.01),vec3(0.04,0.05,0.08),dayF);
    vec3 skyN=mix(vec3(0.004,0.006,0.015),vec3(0.06,0.08,0.14),dayF);
    vec3 skyS=skyN*0.8;
    vec3 irrDir=n*0.5+0.5;
    vec3 irr=mix(skyDn,skyUp,irrDir.y);
    irr+=mix(skyS,skyN,irrDir.z)*0.3;
    vec3 ambient=mat.albedo*(1.0-mat.metallic)*irr*occ;
    // Env BRDF (Karis analytic + multi-scatter GGX)
    float NdV=max(dot(n,V),0.001);
    float fresT=1.0-NdV;float fresT2=fresT*fresT;float fres=fresT2*fresT2*fresT;
    float r=mat.roughness;float r2=r*r;float r4=r2*r2;
    float ebScale=SAT(1.0-max(r-0.04,0.0)*fres-r4*(1.0-NdV));
    ebScale=SAT(ebScale+(1.0-ebScale)*fres);
    float ebBias=fres*SAT(1.0-r2)+r4*0.04;
    vec2 envBRDF=vec2(ebScale,ebBias);
    vec3 specEnv=F0*envBRDF.x+(1.0-F0)*envBRDF.y;
    vec3 reflDir=reflect(-V,n);
    vec3 reflIrr=mix(skyDn,skyUp,reflDir.y*0.5+0.5)+mix(skyS,skyN,reflDir.z*0.5+0.5)*0.25;
    reflIrr=mix(reflIrr,irr*0.6,r);
    ambient+=specEnv*reflIrr*occ;
    // Multi-scatter GGX energy compensation
    float Ems=SAT(1.0-(envBRDF.x+envBRDF.y));
    vec3 Favg=F0+(1.0-F0)*0.047619;
    vec3 msEnergy=Favg*Ems/(1.0-Favg*Ems+0.0001);
    ambient+=msEnergy*irr*occ*0.25;

    // SSS
    if(mat.sss>0.01){{
      float bl=SAT(dot(n,-keyDir))*0.5+0.5;float ts=SAT(sdf_map(p+keyDir*0.3).x*5.0);
      Lo+=mat.albedo*keyCol*bl*ts*mat.sss*0.12;
    }}

    float emBoost=mix(1.3,0.5,dayF);
    vec3 rim=specEnv*fres*0.2;
    col=Lo*occ+ambient+mat.emission*emBoost+rim;
{shatter_cracks}
{ssr}

    // Rain wetness
    if(uWxRain>0.01){{mat.roughness*=mix(1.0,0.04,uWxRain*0.55);}}
  }}
{destruction_atmosphere}

  // Atmosphere fog
  float weatherFogMul=1.0+uWxFog*6.0+uWxRain*2.0;
  float distFog=1.0-exp(-t*0.007*weatherFogMul);
  vec3 hitP=ro+rd*t;
  float heightFog=exp(-max(hitP.y,0.0)*0.12)*0.35*(1.0-exp(-t*0.015));
  float totalFog=SAT(distFog+heightFog);
  vec3 fogCol=mix(mix(vec3(0.005,0.008,0.025),vec3(0.2,0.22,0.28),dayF),sky*0.4,0.25);
  col=mix(col,fogCol,totalFog);
{vol_light}
{rain_streaks}

{post}
  gl_FragColor=vec4(col,1);
}}
",
        weather_uniforms = weather_uniforms,
        destruction_defines = destruction_defines,
        camera_shake = camera_shake,
        day_night = day_night,
        bloom_accum = bloom_accum,
        max_steps = config.max_steps,
        max_dist = format!("{:.1}", config.max_distance),
        bloom_in_loop = bloom_in_loop,
        bloom_to_col = bloom_to_col,
        get_mat = get_mat,
        rain_occ = rain_occ,
        micro_normal = micro_normal_apply,
        shatter_cracks = shatter_cracks,
        ssr = ssr,
        destruction_atmosphere = destruction_atmosphere,
        vol_light = vol_light,
        rain_streaks = rain_streaks,
        post = post,
    )
}

/// SDF map関数 + フルレンダリングパイプラインを結合したGLSLを生成
pub fn build_full_shader(sdf_eval_source: &str, config: &RenderConfig) -> String {
    let version = config.glsl_version;
    let biome = if config.biome_terrain {
        BIOME_SYSTEM
    } else {
        ""
    };
    let vol = if config.volumetric_light {
        VOLUMETRIC_LIGHT
    } else {
        ""
    };
    let post = if config.post_process {
        POST_PROCESS
    } else {
        ""
    };
    let destr_uniforms = if config.destruction {
        DESTRUCTION_UNIFORMS
    } else {
        ""
    };
    let spectral = if config.spectral_rendering {
        SPECTRAL_LIB
    } else {
        ""
    };
    let vfx = if config.vfx_effects {
        VFX_LIB
    } else {
        ""
    };
    let micro = if config.micro_normal {
        MICRO_NORMAL_LIB
    } else {
        ""
    };
    let interior = if config.interior_mapping {
        INTERIOR_MAPPING_LIB
    } else {
        ""
    };
    let destruction = if config.destruction {
        DESTRUCTION_SYSTEM
    } else {
        ""
    };
    let sdf_map_wrapper = if config.material_slots > 1 {
        // マルチマテリアル: sdf_eval が vec2(dist, id) を返す前提
        "\n// sdf_map wrapper: multi-material (sdf_eval returns vec2)\nvec2 sdf_map(vec3 p){\n  return sdf_eval(p);\n}\n".to_string()
    } else {
        "\n// sdf_map wrapper: returns vec2(distance, material_id)\nvec2 sdf_map(vec3 p){\n  return vec2(sdf_eval(p), 0.0);\n}\n".to_string()
    };
    let dual_sdf_section = if config.dual_sdf {
        // map_lite: AO/Shadow/Rain用軽量SDF (ユーザーが sdf_eval_lite を定義する前提)
        "\n// ═══ Dual SDF: map_lite for AO/Shadow/Rain ═══\nfloat sdf_map_lite(vec3 p){\n  return sdf_eval_lite(p);\n}\n"
    } else {
        // map_lite がない場合は sdf_map を使う
        "\nfloat sdf_map_lite(vec3 p){\n  return sdf_map(p).x;\n}\n"
    };

    let main_fn = build_main_function(config);

    format!(
        r"#version {version} es

// ═══════════════════════════════════════════════════════
// ALICE-SDF Full Rendering Pipeline
// Generated by alice-sdf v1.4.0
// ═══════════════════════════════════════════════════════

precision highp float;
{uniforms}
{destr_uniforms}
{noise}
{spectral}
{vfx}
{micro}
{interior}
{biome}
{destruction}
// ═══ SDF Scene (generated) ═══
{sdf_source}
{sdf_map_wrapper}
{dual_sdf}
{pbr}
{normal_ao}
{sky}
{vol}
{post}
{main_fn}
",
        version = version,
        uniforms = UNIFORMS,
        destr_uniforms = destr_uniforms,
        noise = NOISE_LIB,
        spectral = spectral,
        vfx = vfx,
        micro = micro,
        interior = interior,
        biome = biome,
        destruction = destruction,
        sdf_source = sdf_eval_source,
        sdf_map_wrapper = sdf_map_wrapper,
        dual_sdf = dual_sdf_section,
        pbr = PBR_BRDF,
        normal_ao = NORMAL_AO_SHADOW,
        sky = SKY_ATMOSPHERE,
        vol = vol,
        post = post,
        main_fn = main_fn,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const DUMMY_SDF: &str = "float sdf_eval(vec3 p){ return length(p) - 1.0; }";
    const DUMMY_SDF_MULTI: &str =
        "vec2 sdf_eval(vec3 p){ return vec2(length(p) - 1.0, 0.0); }";
    const DUMMY_SDF_DUAL: &str = "float sdf_eval(vec3 p){ return length(p) - 1.0; }\nfloat sdf_eval_lite(vec3 p){ return length(p) - 1.0; }";

    #[test]
    fn default_config_generates_valid_shader() {
        let config = RenderConfig::default();
        let shader = build_full_shader(DUMMY_SDF, &config);
        assert!(shader.contains("void main()"));
        assert!(shader.contains("sdf_eval"));
        assert!(shader.contains("sdf_map"));
        assert!(shader.contains("getDefaultMat"));
        assert!(!shader.contains("blackbody"));
        assert!(!shader.contains("uniform float uShatter"));
        assert!(!shader.contains("dWarp"));
    }

    #[test]
    fn spectral_rendering_adds_blackbody() {
        let config = RenderConfig {
            spectral_rendering: true,
            ..Default::default()
        };
        let shader = build_full_shader(DUMMY_SDF, &config);
        assert!(shader.contains("blackbody"));
        assert!(shader.contains("spectralToRGB"));
        assert!(shader.contains("spectralBlackbody"));
        assert!(shader.contains("rayleighSpectral"));
    }

    #[test]
    fn vfx_effects_adds_domain_warp_and_dbm() {
        let config = RenderConfig {
            vfx_effects: true,
            ..Default::default()
        };
        let shader = build_full_shader(DUMMY_SDF, &config);
        assert!(shader.contains("dWarp"));
        assert!(shader.contains("dbmDischarge"));
        assert!(shader.contains("aBloom"));
        assert!(shader.contains("bloomAcc"));
    }

    #[test]
    fn destruction_adds_voronoi_and_uniforms() {
        let config = RenderConfig {
            destruction: true,
            ..Default::default()
        };
        let shader = build_full_shader(DUMMY_SDF, &config);
        assert!(shader.contains("voronoi2"));
        assert!(shader.contains("destructionAt"));
        assert!(shader.contains("sdDestruction"));
        assert!(shader.contains("sdDebris"));
        assert!(shader.contains("uniform float uShatter"));
        assert!(shader.contains("uniform float uEntropy"));
        assert!(shader.contains("uShake"));
        assert!(shader.contains("uMeteorImpact"));
        assert!(shader.contains("Impact flash"));
    }

    #[test]
    fn micro_normal_adds_function() {
        let config = RenderConfig {
            micro_normal: true,
            ..Default::default()
        };
        let shader = build_full_shader(DUMMY_SDF, &config);
        assert!(shader.contains("microNormal"));
    }

    #[test]
    fn interior_mapping_adds_function() {
        let config = RenderConfig {
            interior_mapping: true,
            ..Default::default()
        };
        let shader = build_full_shader(DUMMY_SDF, &config);
        assert!(shader.contains("interiorMap"));
    }

    #[test]
    fn multi_material_uses_vec2_wrapper() {
        let config = RenderConfig {
            material_slots: 4,
            ..Default::default()
        };
        let shader = build_full_shader(DUMMY_SDF_MULTI, &config);
        assert!(shader.contains("multi-material"));
        assert!(shader.contains("return sdf_eval(p)"));
        assert!(shader.contains("getMat(hit.y,p)"));
    }

    #[test]
    fn dual_sdf_uses_lite_wrapper() {
        let config = RenderConfig {
            dual_sdf: true,
            ..Default::default()
        };
        let shader = build_full_shader(DUMMY_SDF_DUAL, &config);
        assert!(shader.contains("sdf_eval_lite"));
        assert!(shader.contains("sdf_map_lite"));
    }

    #[test]
    fn ssr_enabled_adds_floor_reflection() {
        let config = RenderConfig {
            ssr_enabled: true,
            ..Default::default()
        };
        let shader = build_full_shader(DUMMY_SDF, &config);
        assert!(shader.contains("Floor SSR"));
    }

    #[test]
    fn all_features_combined() {
        let config = RenderConfig {
            spectral_rendering: true,
            material_slots: 8,
            destruction: true,
            vfx_effects: true,
            interior_mapping: true,
            micro_normal: true,
            dual_sdf: true,
            ..Default::default()
        };
        let shader = build_full_shader(DUMMY_SDF_DUAL, &config);
        assert!(shader.contains("blackbody"));
        assert!(shader.contains("dWarp"));
        assert!(shader.contains("voronoi2"));
        assert!(shader.contains("microNormal"));
        assert!(shader.contains("interiorMap"));
        assert!(shader.contains("sdf_eval_lite"));
        assert!(shader.contains("uShatter"));
        assert!(shader.contains("bloomAcc"));
    }

    #[test]
    fn no_features_minimal_shader() {
        let config = RenderConfig {
            day_night_cycle: false,
            weather_system: false,
            ssr_enabled: false,
            volumetric_light: false,
            post_process: false,
            biome_terrain: false,
            spectral_rendering: false,
            destruction: false,
            vfx_effects: false,
            interior_mapping: false,
            micro_normal: false,
            dual_sdf: false,
            ..Default::default()
        };
        let shader = build_full_shader(DUMMY_SDF, &config);
        assert!(shader.contains("void main()"));
        assert!(!shader.contains("blackbody"));
        assert!(!shader.contains("dWarp"));
        assert!(!shader.contains("voronoi2"));
        assert!(!shader.contains("microNormal"));
        assert!(!shader.contains("interiorMap"));
        assert!(!shader.contains("uniform float uShatter"));
        assert!(shader.contains("#define uWxFog 0.0"));
        assert!(shader.contains("#define uShatter 0.0"));
    }
}
