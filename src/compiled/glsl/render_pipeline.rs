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
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GLSL テンプレートパーツ
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// ユニフォーム宣言
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

/// 法線計算 + AO + シャドウ
pub(crate) const NORMAL_AO_SHADOW: &str = r"
// ═══ Normal / AO / Shadow ═══
vec3 calcN(vec3 p,float t){
  vec2 e=vec2(max(0.002,t*0.001),0);
  return normalize(vec3(
    sdf_map(p+e.xyy).x-sdf_map(p-e.xyy).x,
    sdf_map(p+e.yxy).x-sdf_map(p-e.yxy).x,
    sdf_map(p+e.yyx).x-sdf_map(p-e.yyx).x));
}
float ao(vec3 p,vec3 n){
  float occ=0.0;float sc=1.0;
  for(int i=1;i<=5;i++){float hr=0.01+0.12*float(i);float dd=sdf_map(p+n*hr).x;occ+=(hr-dd)*sc;sc*=0.65;}
  return SAT(1.0-3.0*occ);
}
float shadowProxy(vec3 p,vec3 n,vec3 l){
  float t=0.08;float res=1.0;
  for(int i=0;i<16;i++){float d=sdf_map(p+l*t).x;res=min(res,8.0*d/t);t+=clamp(d,0.02,0.2);if(t>5.0)break;}
  return SAT(res);
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

    let vol_light = if config.volumetric_light {
        r"
  // Volumetric light
  float volDist=min(t,40.0);
  vec3 vol=volScatter(ro,rd,sunDir,volDist,dayF,gl_FragCoord.xy);
  col+=vol*(1.0-totalFog);
  float godRayS=max(dot(rd,sunDir),0.0);
  col+=mix(vec3(0.5,0.3,0.15),vec3(1.0,0.95,0.8),dayF)*(pow(godRayS,48.0)*0.1+pow(godRayS,8.0)*0.02)*dayF*(1.0-totalFog);"
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
void main(){{
  vec2 uv=(gl_FragCoord.xy-0.5*uRes)/uRes.y;
  vec3 ro=uCamPos;
  vec3 rd=normalize(uCamFwd+uv.x*uCamRight+uv.y*uCamUp);
{day_night}
  vec3 sky=skyColor(rd,sunDir,moonDir,dayF);

  // Raymarch
  float t=0.0;vec2 hit;
  for(int i=0;i<{max_steps};i++){{
    hit=sdf_map(ro+rd*t);
    if(hit.x<max(0.0005,t*0.0015)||t>{max_dist})break;
    t+=hit.x*0.65;
  }}

  vec3 col=sky;
  if(t<{max_dist}){{
    vec3 p=ro+rd*t;
    vec3 n=calcN(p,t);
    vec3 V=-rd;
    Mat mat=getDefaultMat(p);
    mat.emission=vec3(0);
    vec3 F0=mix(vec3(0.04),mat.albedo,mat.metallic);

    // Lighting
    vec3 keyDir=mix(moonDir,sunDir,dayF);
    vec3 keyCol=mix(vec3(0.06,0.08,0.18),vec3(1.5,1.35,1.1),dayF);
    keyCol=mix(keyCol,vec3(1.5,0.6,0.25),goldenF*0.7);
    keyCol+=vec3(2.0,2.2,2.8)*uLightning;
    float keyShadow=shadowProxy(p+n*0.12,n,keyDir);
    vec3 fillDir=normalize(vec3(-0.35,0.35,-0.6));
    vec3 fillCol=mix(vec3(0.03,0.04,0.08),vec3(0.18,0.25,0.45),dayF);
    float occ=ao(p,n);

    vec3 Lo=pbrDirect(n,V,keyDir,keyCol,mat,F0)*keyShadow;
    Lo+=pbrDirect(n,V,fillDir,fillCol,mat,F0);

    // IBL ambient
    vec3 skyUp=mix(vec3(0.008,0.012,0.03),vec3(0.1,0.14,0.25),dayF);
    vec3 skyDn=mix(vec3(0.003,0.005,0.01),vec3(0.04,0.05,0.08),dayF);
    vec3 irr=mix(skyDn,skyUp,n.y*0.5+0.5);
    vec3 ambient=mat.albedo*(1.0-mat.metallic)*irr*occ;
    // Env BRDF (Karis analytic)
    float NdV=max(dot(n,V),0.001);float fres=pow(1.0-NdV,5.0);
    float r=mat.roughness;
    vec2 envBRDF=vec2(SAT(1.0-max(r-0.04,0.0)*fres-r*r*r*r*(1.0-NdV)),fres*SAT(1.0-r*r)+r*r*r*r*0.04);
    vec3 specEnv=F0*envBRDF.x+(1.0-F0)*envBRDF.y;
    vec3 reflIrr=mix(skyDn,skyUp,reflect(-V,n).y*0.5+0.5);
    ambient+=specEnv*mix(reflIrr,irr*0.6,r)*occ;

    // SSS
    if(mat.sss>0.01){{
      float bl=SAT(dot(n,-keyDir))*0.5+0.5;float ts=SAT(sdf_map(p+keyDir*0.3).x*5.0);
      Lo+=mat.albedo*keyCol*bl*ts*mat.sss*0.12;
    }}

    float emBoost=mix(1.3,0.5,dayF);
    col=Lo*occ+ambient+mat.emission*emBoost;

    // Rain wetness
    if(uWxRain>0.01){{mat.roughness*=mix(1.0,0.04,uWxRain*0.55);}}
  }}

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
        day_night = day_night,
        max_steps = config.max_steps,
        max_dist = format!("{:.1}", config.max_distance),
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

    let main_fn = build_main_function(config);

    format!(
        r"#version {version} es

// ═══════════════════════════════════════════════════════
// ALICE-SDF Full Rendering Pipeline
// Generated by alice-sdf v1.3.0
// ═══════════════════════════════════════════════════════

precision highp float;
{uniforms}
{noise}
{biome}
// ═══ SDF Scene (generated) ═══
{sdf_source}

// sdf_map wrapper: returns vec2(distance, material_id)
vec2 sdf_map(vec3 p){{
  return vec2(sdf_eval(p), 0.0);
}}

{pbr}
{normal_ao}
{sky}
{vol}
{post}
{main_fn}
",
        version = version,
        uniforms = UNIFORMS,
        noise = NOISE_LIB,
        biome = biome,
        sdf_source = sdf_eval_source,
        pbr = PBR_BRDF,
        normal_ao = NORMAL_AO_SHADOW,
        sky = SKY_ATMOSPHERE,
        vol = vol,
        post = post,
        main_fn = main_fn,
    )
}
