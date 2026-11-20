// Coordinates d=t-s and s put all undivided rays through the same opening.
// At fixed d the four source and four receiver hats are quadratic in s.
// Three Gauss values integrate their product analytically. Across d the
// product, interval width and |d| have total degree at most six. Contract
// that polynomial with the rational kernel's integrated Legendre moments.
export const lineMomentsVertex = `#version 300 es
precision highp float;
layout(location=0) in vec4 difference; // centre, half-length, plane gap, normalized angular weight
layout(location=1) in vec4 bounds; // source lower,upper at each end of the difference interval
layout(location=2) in vec4 emitting; // midpoint UV and their physical derivatives along the line
layout(location=3) in vec4 receiving;
out mat4 measure;
const float X[7]=float[7](-.9491079123,-.7415311856,-.4058451514,0.,.4058451514,.7415311856,.9491079123);
const float W[7]=float[7](.1294849662,.2797053915,.3818300505,.4179591837,.3818300505,.2797053915,.1294849662);
vec4 hats(vec4 affine,float t){
  vec2 uv=clamp(affine.xy+t*affine.zw,vec2(0.),vec2(1.));
  vec2 u=vec2(1.-uv.x,uv.x),v=vec2(1.-uv.y,uv.y);
  return u.xyxy*v.xxyy;
}
void main(){
  float d=difference.x,w=difference.y,h=difference.z,den=h*h+d*d;
  float a=2.*d*w/den,b=w*w/den;
  float f=1.,prev=0.,L[7];for(int k=0;k<7;k++)L[k]=0.;
  int order=b<.000256?8:(b<.0025?12:(b<.0144?16:20));
  for(int j=0;j<=20;j++){
    // Integral of z^j P_k(z). Orthogonal modes vanish for k>j or odd j-k.
    float v=2./float(j+1+(j&1));
    for(int k=(j&1);k<7&&k<=j;k+=2){
      L[k]+=f*v;v*=float(j-k)/float(j+k+3);
    }
    if(j==order)break;
    float next=-(float(j+2)*a*f+float(j+3)*b*prev)/float(j+1);
    prev=f;f=next;
  }
  mat4 total=mat4(0.);
  for(int i=0;i<7;i++){
    float q=X[i],last=1.,v=q,coef=L[0]+3.*q*L[1];
    for(int k=2;k<7;k++){
      float p=(float(2*k-1)*q*v-float(k-1)*last)/float(k);
      coef+=float(2*k+1)*p*L[k];last=v;v=p;
    }
    vec2 limits=mix(bounds.xy,bounds.zw,.5+.5*q);
    float m=.5*(limits.y+limits.x),halfLength=.5*(limits.y-limits.x);
    mat4 along=mat4(0.);
    for(int k=-1;k<=1;k++){
      float s=m+float(k)*.7745966692*halfLength;
      vec4 A=hats(emitting,s), B=hats(receiving,s+w*q);
      along+=(k==0?8./9.:5./9.)*outerProduct(A,B);
    }
    total+=(.5*W[i]*coef*abs(d+w*q)*halfLength)*along;
  }
  measure=(difference.w*w*(h/den)*(h/den))*total;
  gl_Position=vec4(0,0,0,1);gl_PointSize=1.;
}`;
