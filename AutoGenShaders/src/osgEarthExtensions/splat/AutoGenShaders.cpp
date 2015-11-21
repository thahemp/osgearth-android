// CMake will compile this file into AutoGenShaders.cpp

#include <osgEarthExtensions/splat/SplatShaders>

#define MULTILINE(...) #__VA_ARGS__

using namespace osgEarth::Splat;

Shaders::Shaders()
{
    Types = "Splat.types.glsl";
    _sources[Types] = MULTILINE(// begin: Splat.types.glsl\n
\n
// Environment structure passed around locally.\n
struct oe_SplatEnv {\n
    float range;\n
    float elevation;\n
    float slope;\n
    vec4 noise;\n
};\n
\n
// Rendering parameters for splat texture and noise-based detail texture.\n
struct oe_SplatRenderInfo {\n
    float primaryIndex;\n
    float detailIndex;\n
    float brightness;\n
    float contrast;\n
    float threshold;\n
    float minSlope;\n
};\n
\n
// end: Splat.types.glsl\n
);

    Noise = "Splat.Noise.glsl";
    _sources[Noise] = MULTILINE($__HASHTAG__version 110\n
//\n
// Description : Array and textureless GLSL 2D/3D/4D simplex \n
//               noise functions.\n
//      Author : Ian McEwan, Ashima Arts.\n
//  Maintainer : ijm\n
//     Lastmod : 20110822 (ijm)\n
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.\n
//               Distributed under the MIT License. See LICENSE file.\n
//               https://github.com/ashima/webgl-noise\n
// \n
\n
vec4 oe_noise_mod289(vec4 x) {\n
  return x - floor(x * (1.0 / 289.0)) * 289.0; }\n
\n
float oe_noise_mod289(float x) {\n
  return x - floor(x * (1.0 / 289.0)) * 289.0; }\n
\n
vec4 oe_noise_permute(vec4 x) {\n
     return oe_noise_mod289(((x*34.0)+1.0)*x);\n
}\n
\n
float oe_noise_permute(float x) {\n
     return oe_noise_mod289(((x*34.0)+1.0)*x);\n
}\n
\n
vec4 oe_noise_taylorInvSqrt(vec4 r)\n
{\n
  return 1.79284291400159 - 0.85373472095314 * r;\n
}\n
\n
float oe_noise_taylorInvSqrt(float r)\n
{\n
  return 1.79284291400159 - 0.85373472095314 * r;\n
}\n
\n
vec4 oe_noise_grad4(float j, vec4 ip)\n
  {\n
  const vec4 ones = vec4(1.0, 1.0, 1.0, -1.0);\n
  vec4 p,s;\n
\n
  p.xyz = floor( fract (vec3(j) * ip.xyz) * 7.0) * ip.z - 1.0;\n
  p.w = 1.5 - dot(abs(p.xyz), ones.xyz);\n
  s = vec4(lessThan(p, vec4(0.0)));\n
  p.xyz = p.xyz + (s.xyz*2.0 - 1.0) * s.www; \n
\n
  return p;\n
  }\n
						\n
// (sqrt(5) - 1)/4 = F4, used once below\n
$__HASHTAG__define oe_noise_F4 0.309016994374947451\n
\n
float oe_noise_snoise(vec4 v)\n
  {\n
  const vec4  C = vec4( 0.138196601125011,  // (5 - sqrt(5))/20  G4\n
                        0.276393202250021,  // 2 * G4\n
                        0.414589803375032,  // 3 * G4\n
                       -0.447213595499958); // -1 + 4 * G4\n
\n
// First corner\n
  vec4 i  = floor(v + dot(v, vec4(oe_noise_F4)) );\n
  vec4 x0 = v -   i + dot(i, C.xxxx);\n
\n
// Other corners\n
\n
// Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)\n
  vec4 i0;\n
  vec3 isX = step( x0.yzw, x0.xxx );\n
  vec3 isYZ = step( x0.zww, x0.yyz );\n
//  i0.x = dot( isX, vec3( 1.0 ) );\n
  i0.x = isX.x + isX.y + isX.z;\n
  i0.yzw = 1.0 - isX;\n
//  i0.y += dot( isYZ.xy, vec2( 1.0 ) );\n
  i0.y += isYZ.x + isYZ.y;\n
  i0.zw += 1.0 - isYZ.xy;\n
  i0.z += isYZ.z;\n
  i0.w += 1.0 - isYZ.z;\n
\n
  // i0 now contains the unique values 0,1,2,3 in each channel\n
  vec4 i3 = clamp( i0, 0.0, 1.0 );\n
  vec4 i2 = clamp( i0-1.0, 0.0, 1.0 );\n
  vec4 i1 = clamp( i0-2.0, 0.0, 1.0 );\n
\n
  //  x0 = x0 - 0.0 + 0.0 * C.xxxx\n
  //  x1 = x0 - i1  + 1.0 * C.xxxx\n
  //  x2 = x0 - i2  + 2.0 * C.xxxx\n
  //  x3 = x0 - i3  + 3.0 * C.xxxx\n
  //  x4 = x0 - 1.0 + 4.0 * C.xxxx\n
  vec4 x1 = x0 - i1 + C.xxxx;\n
  vec4 x2 = x0 - i2 + C.yyyy;\n
  vec4 x3 = x0 - i3 + C.zzzz;\n
  vec4 x4 = x0 + C.wwww;\n
\n
// Permutations\n
  i = oe_noise_mod289(i); \n
  float j0 = oe_noise_permute( oe_noise_permute( oe_noise_permute( oe_noise_permute(i.w) + i.z) + i.y) + i.x);\n
  vec4 j1 = oe_noise_permute( oe_noise_permute( oe_noise_permute( oe_noise_permute (\n
             i.w + vec4(i1.w, i2.w, i3.w, 1.0 ))\n
           + i.z + vec4(i1.z, i2.z, i3.z, 1.0 ))\n
           + i.y + vec4(i1.y, i2.y, i3.y, 1.0 ))\n
           + i.x + vec4(i1.x, i2.x, i3.x, 1.0 ));\n
\n
// Gradients: 7x7x6 points over a cube, mapped onto a 4-cross polytope\n
// 7*7*6 = 294, which is close to the ring size 17*17 = 289.\n
  vec4 ip = vec4(1.0/294.0, 1.0/49.0, 1.0/7.0, 0.0) ;\n
\n
  vec4 p0 = oe_noise_grad4(j0,   ip);\n
  vec4 p1 = oe_noise_grad4(j1.x, ip);\n
  vec4 p2 = oe_noise_grad4(j1.y, ip);\n
  vec4 p3 = oe_noise_grad4(j1.z, ip);\n
  vec4 p4 = oe_noise_grad4(j1.w, ip);\n
\n
// Normalise gradients\n
  vec4 norm = oe_noise_taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));\n
  p0 *= norm.x;\n
  p1 *= norm.y;\n
  p2 *= norm.z;\n
  p3 *= norm.w;\n
  p4 *= oe_noise_taylorInvSqrt(dot(p4,p4));\n
\n
// Mix contributions from the five corners\n
  vec3 m0 = max(0.6 - vec3(dot(x0,x0), dot(x1,x1), dot(x2,x2)), 0.0);\n
  vec2 m1 = max(0.6 - vec2(dot(x3,x3), dot(x4,x4)            ), 0.0);\n
  m0 = m0 * m0;\n
  m1 = m1 * m1;\n
  return 49.0 * ( dot(m0*m0, vec3( dot( p0, x0 ), dot( p1, x1 ), dot( p2, x2 )))\n
               + dot(m1*m1, vec2( dot( p3, x3 ), dot( p4, x4 ) ) ) ) ;\n
}\n
\n
// Generates a tiled fractal simplex noise value and clamps the values to [0..1].\n
float oe_noise_fractal4D(in vec2 seed, in float frequency, in float persistence, in float lacunarity, in int octaves)\n
{\n
    const float TwoPI = 6.283185;\n
	float f = frequency;\n
	float amp = 1.0;\n
	float maxAmp = 0.0;\n
	float n = 0.0;\n
    \n
    vec4 seed4D;\n
    seed4D.xy = cos(seed*TwoPI)/TwoPI;\n
    seed4D.zw = sin(seed*TwoPI)/TwoPI;\n
\n
	for(int i=0; i<octaves; ++i)\n
	{\n
		n += oe_noise_snoise(seed4D*f) * amp;\n
		maxAmp += amp;\n
		amp *= persistence;\n
		f *= lacunarity;\n
	}\n
	//return n / maxAmp;\n
    const float low = 0.0;\n
    const float high = 1.0;\n
\n
    n /= maxAmp;\n
    n = n * (high-low)/2.0 + (high+low)/2.0;\n
    return clamp(n, 0.0, 1.0);\n
}\n
\n
);

    VertModel = "Splat.vert.model.glsl";
    _sources[VertModel] = MULTILINE($__HASHTAG__version 330\n
$__HASHTAG__pragma vp_entryPoint "oe_splat_vertex_model"\n
$__HASHTAG__pragma vp_location   "vertex_model"\n
$__HASHTAG__pragma vp_order      "0.5"\n
\n
out vec3 vp_Normal;\n
out float oe_splat_slope;\n
\n
void oe_splat_vertex_model(inout vec4 VertexMODEL)\n
{\n
    // calculate slope from the Z component of the current normal\n
    // since the terrain is in LTP space.\n
    oe_splat_slope = 1.0-vp_Normal.z;\n
}\n
\n
);

    VertView = "Splat.vert.view.glsl";
    _sources[VertView] = MULTILINE($__HASHTAG__version 330\n
\n
$__HASHTAG__pragma vp_entryPoint "oe_splat_vertex_view"\n
$__HASHTAG__pragma vp_location   "vertex_view"\n
$__HASHTAG__pragma vp_order      "0.5"\n
\n
$__HASHTAG__pragma include "Splat.types.glsl"\n
\n
out vec4 oe_layer_tilec;\n
out float oe_splat_range;\n
out vec2 oe_splat_covtc;\n
\n
uniform mat4 $COVERAGE_TEXMAT_UNIFORM;   // assigned at runtime\n
\n
\n
void oe_splat_vertex_view(inout vec4 VertexVIEW)\n
{\n
    // range from camera to vertex\n
    oe_splat_range = -VertexVIEW.z;\n
\n
    // calculate the coverage sampling coordinates. The texture matrix accounts\n
    // for any super-sampling that might be in effect for the current LOD.\n
    oe_splat_covtc = ($COVERAGE_TEXMAT_UNIFORM * oe_layer_tilec).st;\n
}\n
\n
);

    Frag = "Splat.frag.glsl";
    _sources[Frag] = MULTILINE($__HASHTAG__version 330\n
\n
$__HASHTAG__pragma vp_entryPoint "oe_splat_complex"\n
$__HASHTAG__pragma vp_location   "fragment_coloring"\n
$__HASHTAG__pragma vp_order      "0.4"                 // before terrain image layers\n
\n
// define to activate 'edit' mode in which uniforms control\n
// the splatting parameters.\n
$__HASHTAG__pragma vp_define "SPLAT_EDIT"\n
\n
// define to activate GPU-generated noise instead of a noise texture.\n
$__HASHTAG__pragma vp_define "SPLAT_GPU_NOISE"\n
\n
// include files\n
$__HASHTAG__pragma include "Splat.types.glsl"\n
$__HASHTAG__pragma include "Splat.frag.common.glsl"\n
\n
// ref: Splat.getRenderInfo.frag.glsl\n
oe_SplatRenderInfo oe_splat_getRenderInfo(in float value, in oe_SplatEnv env);\n
\n
// from: Splat.util.glsl\n
void oe_splat_getLodBlend(in float range, in float baseLOD, out float lod0, out float lod1, out float blend);\n
vec2 oe_splat_getSplatCoords(in vec2 coords, in float lod);\n
\n
// from the terrain engine:\n
in vec4 oe_layer_tilec;\n
uniform vec4 oe_tile_key;\n
\n
// from the vertex shader:\n
in vec2 oe_splat_covtc;\n
in float oe_splat_range;\n
\n
// from SplatTerrainEffect:\n
uniform float oe_splat_warp;\n
uniform float oe_splat_blur;\n
uniform sampler2D oe_splat_coverageTex;\n
uniform sampler2DArray oe_splatTex;\n
uniform float oe_splat_scaleOffset;\n
\n
uniform float oe_splat_detailRange;\n
uniform float oe_splat_noiseScale;\n
uniform float oe_splat_useBilinear; // 1=true, -1=false\n
\n
$__HASHTAG__ifdef SPLAT_EDIT\n
uniform float oe_splat_brightness;\n
uniform float oe_splat_contrast;\n
uniform float oe_splat_threshold;\n
uniform float oe_splat_minSlope;\n
$__HASHTAG__endif\n
\n
// Warps the coverage sampling coordinates to mitigate blockiness.\n
vec2 oe_splat_warpCoverageCoords(in vec2 splat_tc, in oe_SplatEnv env)\n
{\n
    vec2 seed = oe_splat_covtc;\n
    float n1 = 2.0*env.noise.y-1.0;\n
    vec2 tc = oe_splat_covtc + n1*oe_splat_warp;\n
    return clamp(tc, 0.0, 1.0);\n
}\n
\n
vec4 oe_splat_getTexel(in float index, in vec2 tc)\n
{\n
    return texture(oe_splatTex, vec3(tc, index));\n
}\n
\n
// Samples a detail texel using its render info parameters.\n
// Returns the weighting factor in the alpha channel.\n
vec4 oe_splat_getDetailTexel(in oe_SplatRenderInfo ri, in vec2 tc, in oe_SplatEnv env)\n
{\n
    float hasDetail = ri.detailIndex >= 0.0 ? 1.0 : 0.0;\n
\n
$__HASHTAG__ifdef SPLAT_EDIT\n
    float brightness = oe_splat_brightness;\n
    float contrast = oe_splat_contrast;\n
    float threshold = oe_splat_threshold;\n
    float minSlope = oe_splat_minSlope;\n
$__HASHTAG__else\n
    float brightness = ri.brightness;\n
    float contrast = ri.contrast;\n
    float threshold = ri.threshold;\n
    float minSlope = ri.minSlope;\n
$__HASHTAG__endif\n
\n
    // start with the noise value\n
    float n = env.noise.x;\n
	\n
    // apply slope limiter, then reclamp and threshold:\n
    float s;\n
    if ( env.slope >= minSlope )\n
        s = 1.0;\n
    else if ( env.slope < 0.1*minSlope )\n
        s = 0.0;\n
    else\n
        s = (env.slope-0.1*minSlope)/(minSlope-0.1*minSlope);\n
\n
    brightness *= s;\n
\n
    // apply brightness and contrast, then reclamp\n
    n = clamp(((n-0.5)*contrast + 0.5) * brightness, 0.0, 1.0);\n
    \n
    // apply final threshold:\n
	n = n < threshold ? 0.0 : n;\n
\n
    // sample the texel and return it.\n
    vec4 result = oe_splat_getTexel( max(ri.detailIndex,0), tc);\n
    return vec4(result.rgb, hasDetail*n);\n
}\n
\n
// Generates a texel using nearest-neighbor coverage sampling.\n
vec4 oe_splat_nearest(in vec2 splat_tc, in oe_SplatEnv env)\n
{\n
    vec2 tc = oe_splat_covtc; //oe_splat_warpCoverageCoords(splat_tc, env);\n
    float coverageValue = texture2D(oe_splat_coverageTex, tc).r;\n
    oe_SplatRenderInfo ri = oe_splat_getRenderInfo(coverageValue, env);\n
    vec4 primary = oe_splat_getTexel(ri.primaryIndex, splat_tc);\n
    float detailToggle = ri.detailIndex >= 0 ? 1.0 : 0.0;\n
    vec4 detail  = oe_splat_getDetailTexel(ri, splat_tc, env) * detailToggle;    \n
    return vec4( mix(primary.rgb, detail.rgb, detail.a), 1.0 );\n
}\n
\n
// Generates a texel using bilinear filtering on the coverage data.\n
vec4 oe_splat_bilinear(in vec2 splat_tc, in oe_SplatEnv env)\n
{\n
    vec4 texel = vec4(0,0,0,1);\n
\n
    //TODO: coverage warping is slow due to the noise function. Consider removing/reworking.\n
    vec2 tc = oe_splat_covtc; //oe_splat_warpCoverageCoords(splat_tc, env);\n
\n
    float a = oe_splat_blur;\n
    float pixelWidth = a/256.0; // 256 = hard-coded cov tex size //TODO \n
    float halfPixelWidth = 0.5*pixelWidth;\n
    float pixelWidth2 = pixelWidth*pixelWidth;\n
\n
    // Find the four quantized coverage coordinates that form a box around the actual\n
    // coverage coordinates, where each quantized coord is at the center of a coverage texel.\n
    vec2 rem = mod(tc, pixelWidth);\n
    vec2 sw;\n
    sw.x = tc.x - rem.x + (rem.x >= halfPixelWidth ? halfPixelWidth : -halfPixelWidth);\n
    sw.y = tc.y - rem.y + (rem.y >= halfPixelWidth ? halfPixelWidth : -halfPixelWidth);\n
    vec2 ne = sw + pixelWidth;\n
    vec2 nw = vec2(sw.x, ne.y);\n
    vec2 se = vec2(ne.x, sw.y);\n
\n
    // Calculate the weighting for each corner.\n
    vec2 dsw = tc-sw;\n
    vec2 dse = tc-se;\n
    vec2 dne = tc-ne;\n
    vec2 dnw = tc-nw;\n
\n
    float sw_weight = max(pixelWidth2-dot(dsw,dsw),0.0);\n
    float se_weight = max(pixelWidth2-dot(dse,dse),0.0);\n
    float ne_weight = max(pixelWidth2-dot(dne,dne),0.0);\n
    float nw_weight = max(pixelWidth2-dot(dnw,dnw),0.0);\n
\n
    // normalize the weights so they total 1.0\n
    float invTotalWeight = 1.0/(sw_weight+se_weight+ne_weight+nw_weight);\n
    sw_weight *= invTotalWeight;\n
    se_weight *= invTotalWeight;\n
    ne_weight *= invTotalWeight;\n
    nw_weight *= invTotalWeight;\n
\n
    // Sample coverage values using quantized corner coords:\n
    float value_sw = texture2D(oe_splat_coverageTex, clamp(sw, 0.0, 1.0)).r;\n
    float value_se = texture2D(oe_splat_coverageTex, clamp(se, 0.0, 1.0)).r;\n
    float value_ne = texture2D(oe_splat_coverageTex, clamp(ne, 0.0, 1.0)).r;\n
    float value_nw = texture2D(oe_splat_coverageTex, clamp(nw, 0.0, 1.0)).r;\n
\n
    // Build the render info data for each corner:\n
    oe_SplatRenderInfo ri_sw = oe_splat_getRenderInfo(value_sw, env);\n
    oe_SplatRenderInfo ri_se = oe_splat_getRenderInfo(value_se, env);\n
    oe_SplatRenderInfo ri_ne = oe_splat_getRenderInfo(value_ne, env);\n
    oe_SplatRenderInfo ri_nw = oe_splat_getRenderInfo(value_nw, env);\n
\n
    // Primary splat:\n
    vec3 sw_primary = oe_splat_getTexel(ri_sw.primaryIndex, splat_tc).rgb;\n
    vec3 se_primary = oe_splat_getTexel(ri_se.primaryIndex, splat_tc).rgb;\n
    vec3 ne_primary = oe_splat_getTexel(ri_ne.primaryIndex, splat_tc).rgb;\n
    vec3 nw_primary = oe_splat_getTexel(ri_nw.primaryIndex, splat_tc).rgb;\n
\n
    // Detail splat - weighting is in the alpha channel\n
    // TODO: Pointless to have a detail range? -gw\n
    // TODO: If noise is a texture, just try to single-sample it instead\n
    float detailToggle =env.range < oe_splat_detailRange ? 1.0 : 0.0;\n
    vec4 sw_detail = detailToggle * oe_splat_getDetailTexel(ri_sw, splat_tc, env);\n
    vec4 se_detail = detailToggle * oe_splat_getDetailTexel(ri_se, splat_tc, env);\n
    vec4 ne_detail = detailToggle * oe_splat_getDetailTexel(ri_ne, splat_tc, env);\n
    vec4 nw_detail = detailToggle * oe_splat_getDetailTexel(ri_nw, splat_tc, env);   \n
\n
    // Combine everything based on weighting:\n
    texel.rgb =\n
        sw_weight * mix(sw_primary, sw_detail.rgb, sw_detail.a) +\n
        se_weight * mix(se_primary, se_detail.rgb, se_detail.a) +\n
        ne_weight * mix(ne_primary, ne_detail.rgb, ne_detail.a) +\n
        nw_weight * mix(nw_primary, nw_detail.rgb, nw_detail.a);\n
\n
    return texel;\n
}\n
\n
$__HASHTAG__ifdef SPLAT_GPU_NOISE\n
\n
uniform float oe_splat_freq;\n
uniform float oe_splat_pers;\n
uniform float oe_splat_lac;\n
uniform float oe_splat_octaves;\n
\n
// see: Splat.Noise.glsl\n
float oe_noise_fractal4D(in vec2 seed, in float frequency, in float persistence, in float lacunarity, in int octaves);\n
\n
vec4 oe_splat_getNoise(in vec2 tc)\n
{\n
    return vec4(oe_noise_fractal4D(tc, oe_splat_freq, oe_splat_pers, oe_splat_lac, int(oe_splat_octaves)));\n
}\n
\n
$__HASHTAG__else // !SPLAT_GPU_NOISE\n
\n
uniform sampler2D oe_splat_noiseTex;\n
vec4 oe_splat_getNoise(in vec2 tc)\n
{\n
    return texture(oe_splat_noiseTex, tc.st);\n
}\n
\n
$__HASHTAG__endif // SPLAT_GPU_NOISE\n
\n
// Simplified entry point with does no filtering or range blending. (much faster.)\n
void oe_splat_simple(inout vec4 color)\n
{\n
    float noiseLOD = floor(oe_splat_noiseScale);\n
    vec2 noiseCoords = oe_splat_getSplatCoords(oe_layer_tilec.st, noiseLOD);\n
\n
    oe_SplatEnv env;\n
    env.range = oe_splat_range;\n
    env.slope = oe_splat_getSlope();\n
    env.noise = oe_splat_getNoise(noiseCoords);\n
    env.elevation = 0.0;\n
\n
    color = oe_splat_bilinear(oe_layer_tilec.st, env);\n
}\n
\n
// Main entry point for fragment shader.\n
void oe_splat_complex(inout vec4 color)\n
{\n
    // Noise coords.\n
    float noiseLOD = floor(oe_splat_noiseScale);\n
    vec2 noiseCoords = oe_splat_getSplatCoords(oe_layer_tilec.st, noiseLOD); //TODO: move to VS for slight speedup\n
\n
    oe_SplatEnv env;\n
    env.range = oe_splat_range;\n
    env.slope = oe_splat_getSlope();\n
    env.noise = oe_splat_getNoise(noiseCoords);\n
    env.elevation = 0.0;\n
\n
    // quantize the scale offset so we take the hit in the FS\n
    float scaleOffset = oe_splat_scaleOffset >= 0.0 ? ceil(oe_splat_scaleOffset) : floor(oe_splat_scaleOffset);\n
        \n
    // Calculate the 2 LODs we need to blend. We have to do this in the FS because \n
    // it's quite possible for a single triangle to span more than 2 LODs.\n
    float lod0;\n
    float lod1;\n
    float lodBlend = -1.0;\n
    oe_splat_getLodBlend(oe_splat_range, scaleOffset, lod0, lod1, lodBlend);\n
\n
    // Sample the two LODs:\n
    vec2 tc0 = oe_splat_getSplatCoords(oe_layer_tilec.st, lod0);\n
    vec4 texel0 = oe_splat_bilinear(tc0, env);\n
    \n
    vec2 tc1 = oe_splat_getSplatCoords(oe_layer_tilec.st, lod1);\n
    vec4 texel1 = oe_splat_bilinear(tc1, env);\n
    \n
    // Blend:\n
    vec4 texel = mix(texel0, texel1, lodBlend);\n
\n
    color = mix(color, texel, texel.a);\n
\n
    // uncomment to visualize slope.\n
    //color.rgba = vec4(env.slope,0,0,1);\n
}\n
\n
);

    FragCommon = "Splat.frag.common.glsl";
    _sources[FragCommon] = MULTILINE(// begin: Splat.frag.common.glsl\n
\n
$__HASHTAG__pragma vp_define "OE_USE_NORMAL_MAP"\n
$__HASHTAG__ifdef OE_USE_NORMAL_MAP\n
\n
// normal map version:\n
uniform sampler2D oe_nmap_normalTex;\n
in vec4 oe_nmap_normalCoords;\n
\n
float oe_splat_getSlope()\n
{\n
    vec4 encodedNormal = texture2D(oe_nmap_normalTex, oe_nmap_normalCoords.st);\n
    vec3 normalTangent = normalize(encodedNormal.xyz*2.0-1.0);\n
    return clamp((1.0-normalTangent.z)/0.8, 0.0, 1.0);\n
}\n
\n
$__HASHTAG__else // !OE_USE_NORMAL_MAP\n
\n
// non- normal map version:\n
in float oe_splat_slope;\n
\n
float oe_splat_getSlope()\n
{\n
    return oe_splat_slope;\n
}\n
\n
$__HASHTAG__endif // OE_USE_NORMAL_MAP\n
\n
// end: Splat.frag.common.glsl\n
);

    FragGetRenderInfo = "Splat.frag.getRenderInfo.glsl";
    _sources[FragGetRenderInfo] = MULTILINE($__HASHTAG__version 330\n
\n
$__HASHTAG__pragma include "Splat.types.glsl"\n
\n
// Samples the coverage data and returns main and detail indices.\n
oe_SplatRenderInfo oe_splat_getRenderInfo(in float value, in oe_SplatEnv env)\n
{\n
    float primary = -1.0;   // primary texture index\n
    float detail = -1.0;    // detail texture index\n
    float brightness = 1.0; // default noise function brightness factor\n
    float contrast = 1.0;   // default noise function contrast factor\n
    float threshold = 0.0;  // default noise function threshold\n
    float slope = 0.0;      // default minimum slope\n
\n
    $CODE_INJECTION_POINT\n
\n
    return oe_SplatRenderInfo(primary, detail, brightness, contrast, threshold, slope);\n
}\n
\n
);

    Util = "Splat.util.glsl";
    _sources[Util] = MULTILINE($__HASHTAG__version 120\n
$__HASHTAG__pragma vp_location "fragment_coloring"\n
\n
uniform vec4 oe_tile_key;  // osgEarth TileKey\n
\n
\n
// Mapping of view ranges to splat texture levels of detail.\n
$__HASHTAG__define RANGE_COUNT 11\n
const float oe_SplatRanges[RANGE_COUNT] = float[](  50.0, 125.0, 250.0, 500.0, 1000.0, 4000.0, 30000.0, 150000.0, 300000.0, 1000000.0, 5000000.0 );\n
const float oe_SplatLevels[RANGE_COUNT] = float[](  20.0,  19.0,  18.0,  17.0,   16.0,   14.0,    12.0,     10.0,      8.0,       6.0,       4.0 );\n
\n
/**\n
 * Given a camera distance, return the two LODs it falls between and\n
 * the blend factor [0..1] between then.\n
 * in  range   = camera distace to fragment\n
 * in  baseLOD = LOD at which texture scale is 1.0\n
 * out LOD0    = near LOD\n
 * out LOD1    = far LOD\n
 * out blend   = Blend factor between LOD0 and LOD1 [0..1]\n
 */\n
void\n
oe_splat_getLodBlend(in float range, in float baseLOD, out float out_LOD0, out float out_LOD1, out float out_blend)\n
{\n
    float clampedRange = clamp(range, oe_SplatRanges[0], oe_SplatRanges[RANGE_COUNT-1]);\n
\n
    out_blend = -1.0;\n
    for(int i=0; i<RANGE_COUNT-1 && out_blend < 0; ++i)\n
    {\n
        if ( clampedRange >= oe_SplatRanges[i] && clampedRange <= oe_SplatRanges[i+1] )\n
        {\n
            out_LOD0 = oe_SplatLevels[i]   + baseLOD;\n
            out_LOD1 = oe_SplatLevels[i+1] + baseLOD;\n
            out_blend = clamp((clampedRange-oe_SplatRanges[i])/(oe_SplatRanges[i+1]-oe_SplatRanges[i]), 0.0, 1.0);\n
        }\n
    }\n
}\n
\n
/**\n
 * Scales the incoming tile splat coordinates to match the requested\n
 * LOD level. We offset the level from the current tile key's LOD (.z)\n
 * because otherwise you run into single-precision jitter at high LODs.\n
 */\n
vec2 \n
oe_splat_getSplatCoords(in vec2 tc, float lod)\n
{\n
    float dL = oe_tile_key.z - lod;\n
    float factor = exp2(dL);\n
    float invFactor = 1.0/factor;\n
    vec2 scale = vec2(invFactor); \n
    vec2 result = tc * scale;\n
\n
    // For upsampling we need to calculate an offset as well\n
    if ( factor >= 1.0 )\n
    {\n
        vec2 a = floor(oe_tile_key.xy * invFactor);\n
        vec2 b = a * factor;\n
        vec2 c = (a+1.0) * factor;\n
        vec2 offset = (oe_tile_key.xy-b)/(c-b);\n
        result += offset;\n
    }\n
\n
    return result;\n
}\n
\n
\n
);
}
