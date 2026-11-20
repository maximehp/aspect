// The CPU has already resolved the object and lamp visibility for each run.
// The fragment stage evaluates that one authored surface at the pixel center.
export const vertex = `#version 300 es
layout(location=0) in ivec4 run;
layout(location=1) in int state;
uniform vec2 size;
flat out int objectId; flat out int visible;
void main(){
 float x=float(gl_VertexID==0?run.x:run.y);
 gl_Position=vec4(2.*vec2(x,float(run.z)+.5)/size-1.,0,1);
 objectId=run.w;visible=state;
}`;
