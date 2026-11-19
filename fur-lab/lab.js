'use strict';
(async()=>{
const $=id=>document.getElementById(id),canvas=$('view'),status=$('status'),gl=canvas.getContext('webgl2',{alpha:false,antialias:false,preserveDrawingBuffer:true});
if(!gl){status.textContent='WebGL 2 is required for the final image draw.';return;}
const program=gl.createProgram();
for(const [type,src] of [[gl.VERTEX_SHADER,`#version 300 es
out vec2 screen;void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);screen=p*2.-1.;gl_Position=vec4(screen,0,1);}`],[gl.FRAGMENT_SHADER,`#version 300 es
precision highp float;in vec2 screen;uniform vec2 origin,right,up;uniform sampler2D img;out vec4 c;void main(){c=texture(img,origin+right*screen.x+up*screen.y);}`]]){
 const sh=gl.createShader(type);gl.shaderSource(sh,src);gl.compileShader(sh);if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(sh));gl.attachShader(program,sh);
}
gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));gl.useProgram(program);
const loc=Object.fromEntries(['origin','right','up'].map(v=>[v,gl.getUniformLocation(program,v)]));
const tex=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,tex);for(const p of [gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_2D,p,gl.LINEAR);for(const p of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T])gl.texParameteri(gl.TEXTURE_2D,p,gl.CLAMP_TO_EDGE);
const nativeTarget={};let nativeW=0,nativeH=0; const nativeTex=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,nativeTex);for(let p of [gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_2D,p,gl.NEAREST);gl.bindTexture(gl.TEXTURE_2D,tex);
let picked=null;
let a,b,res,texN=0,changed=true,az=-1.38,el=.91,zoom=13,cx=0,cy=0,tool='orbit',moving=false,busy=false,last=0,report,drag=null,frame=0;
function allocate(){res=+$('detail').value;a=Coat.makeAtlas(res);b=Coat.makeAtlas(res);changed=true;}
allocate();
function visit(x,y){
 Coat.locate(x,y);const [gx,gy]=Coat.location();$('col').value=gx+5000;$('row').value=gy+5000;
 cx=cy=0;picked=null;allocate();invalidate();
}
$('visit').onclick=()=>visit(Number($('col').value)-5000,Number($('row').value)-5000);
$('edge').onclick=()=>visit(4999,4999);
function matrices(){let c=Math.cos(az),s=Math.sin(az),aspect=canvas.height/canvas.width/Math.sin(el);
 const shift=1.7/Math.tan(el);
 const limit=Math.min((19.9-Math.abs(cx-c*shift))/(Math.abs(s)+Math.abs(c)*aspect),(19.9-Math.abs(cy-s*shift))/(Math.abs(c)+Math.abs(s)*aspect));
 const span=Math.min(zoom,limit),k=span*aspect;
 return [[cx-c*1.7/Math.tan(el),cy-s*1.7/Math.tan(el)],[-s*span,c*span],[-c*k,-s*k]];}
function paint(pixels,upload){
 const m=matrices(),E=Coat.extent*2;for(const [i,k] of ['origin','right','up'].entries())gl.uniform2f(loc[k],m[i][0]/E+(i?0:.5),m[i][1]/E+(i?0:.5));
 if(upload){const bytes=new Uint8Array(pixels.buffer);if(texN!==a.n){gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,a.n,a.n,0,gl.RGBA,gl.UNSIGNED_BYTE,bytes);texN=a.n;}else gl.texSubImage2D(gl.TEXTURE_2D,0,0,0,a.n,a.n,gl.RGBA,gl.UNSIGNED_BYTE,bytes);}
 gl.viewport(0,0,canvas.width,canvas.height);gl.drawArrays(gl.TRIANGLES,0,3);
}
function paintNative(pixels){
 const w=canvas.width,h=canvas.height;gl.bindTexture(gl.TEXTURE_2D,nativeTex);
 const bytes=new Uint8Array(pixels.buffer);
 if(w!==nativeW||h!==nativeH){gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,w,h,0,gl.RGBA,gl.UNSIGNED_BYTE,bytes);nativeW=w;nativeH=h;}
 else gl.texSubImage2D(gl.TEXTURE_2D,0,0,0,w,h,gl.RGBA,gl.UNSIGNED_BYTE,bytes);
 gl.uniform2f(loc.origin,.5,.5);gl.uniform2f(loc.right,.5,0);gl.uniform2f(loc.up,0,.5);gl.viewport(0,0,w,h);gl.drawArrays(gl.TRIANGLES,0,3);gl.bindTexture(gl.TEXTURE_2D,tex);
}
function render(mode,force=true){
 let runs=null,pix=a.base,stats='';
 if(mode==='native'||mode==='heat'){
   runs=Coat.prepare(az,el);const w=canvas.width,h=canvas.height;
   if(mode==='heat'){Coat.render(runs,a,false);paint(a.pixels,true);}
   let pixels=Coat.native(runs,matrices(),w,h,nativeTarget);
   if(mode==='heat'){let read=new Uint8Array(w*h*4);gl.readPixels(0,0,w,h,gl.RGBA,gl.UNSIGNED_BYTE,read);for(let i=0;i<w*h;i++){let c=pixels[i],d=0;for(let k=0;k<3;k++)d=Math.max(d,Math.abs(read[4*i+k]-((c>>>(8*k))&255)));pixels[i]=(0xff000000|(Math.min(255,d*4)<<16)|(Math.min(255,d*5)))>>>0;}}
   paintNative(pixels);return {runs,stats};}
 if(force&&mode!=='bare'){runs=Coat.prepare(az,el);
  if(mode!=='reference')Coat.render(runs,a,false);
  if(mode!=='model')Coat.render(runs,b,true);
  if(mode==='difference'){let bad=0;for(let i=0;i<a.pixels.length;i++){if(a.pixels[i]!==b.pixels[i]){bad++;a.pixels[i]=0xffaa22ff;}else a.pixels[i]=0xff242420;}stats=' · '+bad+' differing atlas samples';}
  pix=mode==='reference'?b.pixels:a.pixels;
 }
 paint(pix,force);return {runs,stats};
}
function refresh(){if(busy)return;if(!frame)frame=requestAnimationFrame(()=>{frame=0;let r=canvas.getBoundingClientRect(),w=Math.max(2,Math.min(1920,Math.round(r.width*devicePixelRatio))),h=Math.round(w*9/16);if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
 let data=render($('mode').value,changed);changed=false;status.textContent=Coat.count.toLocaleString('en-US')+' rooted fibers · patch ID '+((Coat.location()[1]+5000)*10000+Coat.location()[0]+5000)+' · '+Coat.edited()+' brushed · '+Math.round(el*180/Math.PI)+'° elevation'+(picked===null?'':' · touched hair '+picked)+data.stats+(report?'\nLast 4K run: '+report.summary:'');});}
function invalidate(){changed=true;refresh();}
function setTool(t){tool=t;canvas.classList.toggle('brush',t==='brush');for(const k of ['orbit','brush'])$(k).setAttribute('aria-pressed',String(k===t));}
$('orbit').onclick=()=>setTool('orbit');$('brush').onclick=()=>setTool('brush');$('home').onclick=()=>{az=-1.38;el=.91;zoom=13;visit(0,0);};$('part').onclick=()=>{Coat.part();invalidate();};$('reset').onclick=()=>{Coat.reset();invalidate();};$('mode').onchange=invalidate;$('detail').onchange=()=>{allocate();invalidate();};
function ground(e,contact=false){const r=canvas.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*2-1,y=1-(e.clientY-r.top)/r.height*2,m=matrices();const p=[0,1].map(k=>m[0][k]+x*m[1][k]+y*m[2][k]);if(contact){const s=Coat.sample(Coat.prepare(az,el),p[0],p[1],true);if(s){picked=s.rootID;p[0]+=s.z*Math.cos(az)/Math.tan(el);p[1]+=s.z*Math.sin(az)/Math.tan(el);}}return p;}
canvas.onpointerdown=e=>{if(busy)return;canvas.focus();canvas.setPointerCapture(e.pointerId);drag={x:e.clientX,y:e.clientY,p:ground(e,tool==='brush'&&!e.shiftKey),pan:e.shiftKey};};
canvas.onpointermove=e=>{if(!drag||busy)return;let p=ground(e,tool==='brush'&&!drag.pan),dx=e.clientX-drag.x,dy=e.clientY-drag.y;
 if(drag.pan){cx+=drag.p[0]-p[0];cy+=drag.p[1]-p[1];
 if(Math.abs(cx)>=1||Math.abs(cy)>=1){const [x,y]=Coat.location(),u=x+cx,v=y+cy;visit(u,v);const [xx,yy]=Coat.location();cx=Math.max(-1,Math.min(1,u-xx));cy=Math.max(-1,Math.min(1,v-yy));}refresh();}
 else if(tool==='brush'){Coat.brush(...drag.p,...p);invalidate();}
 else{az-=dx*.008;el=Math.max(.43,Math.min(1.48,el+dy*.006));invalidate();}
 drag={x:e.clientX,y:e.clientY,p:ground(e,tool==='brush'&&!drag.pan),pan:drag.pan};};
canvas.onpointerup=canvas.onpointercancel=()=>drag=null;
function magnify(f){zoom=Math.max(2,Math.min(25,zoom*f));refresh();}
canvas.onwheel=e=>{e.preventDefault();if(!busy)magnify(Math.exp(e.deltaY*.001));};
canvas.onkeydown=e=>{if(busy)return;let key=e.key;if(key==='+'||key==='=')magnify(.85);else if(key==='-')magnify(1.18);else if(key.startsWith('Arrow')){e.preventDefault();az+=key==='ArrowRight'?.12:key==='ArrowLeft'?-.12:0;el=Math.max(.43,Math.min(1.48,el+(key==='ArrowUp'?.08:key==='ArrowDown'?-.08:0)));invalidate();}};
function anim(t){if(!moving||busy)return;az+=Math.min(100,t-last)*.00034;last=t;invalidate();requestAnimationFrame(anim);}
$('walk').onclick=()=>{moving=!moving;$('walk').textContent=moving?'Stop walk':'Walk around';if(moving){last=performance.now();requestAnimationFrame(anim);}};
const sync=new Uint8Array(4),tick=()=>new Promise(r=>requestAnimationFrame(r)),median=v=>{const n=v.slice().sort((a,b)=>a-b);return (n[(n.length-1)>>1]+n[n.length>>1])/2;};
async function benchmark(poses=12){
 if(busy)return;busy=true;moving=false;$('walk').textContent='Walk around';drag=null;if(frame){cancelAnimationFrame(frame);frame=0;}
 const saved=[az,el,zoom,cx,cy];for(const e of document.querySelectorAll('button, select'))e.disabled=true;
 try{canvas.width=3840;canvas.height=2160;if(gl.drawingBufferWidth!==3840||gl.drawingBufferHeight!==2160)throw Error('This device could not allocate a 4K drawing buffer.');
 let rows=[];cx=cy=0;
 const measure=mode=>{let start=performance.now(),runs=mode==='bare'?null:Coat.prepare(az,el);
 let pix=a.base;if(mode==='native'){pix=Coat.native(runs,matrices(),3840,2160,nativeTarget);}else if(runs){let target=mode==='model'?a:b;Coat.render(runs,target,mode==='reference');pix=target.pixels;}
 let cpu=performance.now()-start;if(mode==='native')paintNative(pix);else paint(pix,true);gl.readPixels(1920,1080,1,1,gl.RGBA,gl.UNSIGNED_BYTE,sync);
 return {full:performance.now()-start,cpu,runs};};
 // Bare uses a resident backing image. Switching textures has no scene-dependent CPU work.
 let bareTex=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,bareTex);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);for(let p of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T])gl.texParameteri(gl.TEXTURE_2D,p,gl.CLAMP_TO_EDGE);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,a.n,a.n,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array(a.base.buffer));gl.bindTexture(gl.TEXTURE_2D,tex);
 const baseline=()=>{let t=performance.now();gl.bindTexture(gl.TEXTURE_2D,bareTex);paint(null,false);gl.readPixels(0,0,1,1,gl.RGBA,gl.UNSIGNED_BYTE,sync);gl.bindTexture(gl.TEXTURE_2D,tex);return performance.now()-t;};
 measure('model');baseline();measure('reference');measure('native');
 for(let f=0;f<poses;f++){status.textContent='Measuring 4K view '+(f+1)+' of '+poses+'…';await tick();az=-1.38+f*Math.PI*2/poses;el=.83+.32*Math.sin(1.38+az);zoom=8+3*Math.cos(az);
 let bare,model,ref;if(f%2){ref=measure('reference');model=measure('model');bare=baseline();}else{bare=baseline();model=measure('model');ref=measure('reference');}
 let nativeTime=measure('native');let rgb=0,depth=0,worst=0;for(let i=0;i<a.pixels.length;i++){rgb+=a.pixels[i]!==b.pixels[i];if(a.depth[i]!==b.depth[i]){let d=Math.abs(a.depth[i]-b.depth[i]);if(d>1e-7)depth++;worst=Math.max(worst,d);}}
 // Compare the actual filtered 4K model framebuffer with independent strand rays at native pixel centers.
 paint(a.pixels,true);const read=new Uint8Array(3840*2160*4);gl.readPixels(0,0,3840,2160,gl.RGBA,gl.UNSIGNED_BYTE,read);
 let nativeError=0,beyond16=0;for(let p=0;p<nativeTarget.pixels.length;p++){let col=nativeTarget.pixels[p],big=0;for(let k=0;k<3;k++){let d=Math.abs(read[4*p+k]-((col>>>(8*k))&255));nativeError+=d;big=Math.max(big,d);}beyond16+=big>16;}
 let m=matrices(),err=0,over16=0,seed=7919+f;for(let i=0;i<1024;i++){seed=(Math.imul(seed,1664525)+1013904223)>>>0;let x=seed%3840;seed=(Math.imul(seed,1664525)+1013904223)>>>0;let y=seed%2160,sx=(x+.5)/1920-1,sy=(y+.5)/1080-1;
 let col=Coat.sample(model.runs,m[0][0]+m[1][0]*sx+m[2][0]*sy,m[0][1]+m[1][1]*sx+m[2][1]*sy),mx=0;for(let k=0;k<3;k++){let d=Math.abs(((col>>>(8*k))&255)-read[4*(y*3840+x)+k]);err+=d;mx=Math.max(mx,d);}over16+=mx>16;}
 rows.push({pose:f,az,el,zoom,candidateRoots:model.runs.reduce((n,g)=>n+g.last-g.first+1,0),runs:model.runs.length,bare:bare,model:model.full,modelCPU:model.cpu,reference:ref.full,referenceCPU:ref.cpu,native:nativeTime.full,nativeCPU:nativeTime.cpu,nativeMAE:nativeError/(3840*2160*3),nativePercentOver16:100*beyond16/(3840*2160),added:model.full-bare,atlasRGBMismatch:rgb,atlasDepthMismatch:depth,worstDepth:Number.isFinite(worst)?worst:null,screenMAE:err/3072,screenOver16:over16});
 }
 gl.deleteTexture(bareTex);let med=k=>median(rows.map(r=>r[k]));report={resolution:[3840,2160],atlas:a.n,roots:Coat.count,patch:Coat.location(),edited:Coat.edited(),rows,summary:'model '+med('model').toFixed(1)+' ms · individual hairs '+med('native').toFixed(1)+' ms · fur adds '+med('added').toFixed(1)+' ms (paired medians)'};return report;
 }finally{[az,el,zoom,cx,cy]=saved;busy=false;for(const e of document.querySelectorAll('button, select'))e.disabled=false;$('export').disabled=!report;invalidate();}
}
$('bench').onclick=()=>benchmark().catch(e=>{status.textContent=e.message;});
$('export').onclick=()=>{let link=document.createElement('a');link.href=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));link.download='two-tuft-4k.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);};
new ResizeObserver(refresh).observe(canvas);Object.assign(window,{furStudy:{benchmark,visit,part:()=>{Coat.part();invalidate();},reset:()=>{Coat.reset();invalidate();},pose:(angle,elev,z)=>{az=angle;el=elev;zoom=z;invalidate();},info:()=>({roots:Coat.count,edited:Coat.edited(),atlas:a.n})}});refresh();
})().catch(e=>document.getElementById('status').textContent=e.message);
