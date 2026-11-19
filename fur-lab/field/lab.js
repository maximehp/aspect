'use strict';
(()=>{
  const $=id=>document.getElementById(id),canvas=$('view'),status=$('status');
  const gl=canvas.getContext('webgl2',{alpha:false,antialias:false,depth:true,preserveDrawingBuffer:true});
  if(!gl){status.textContent='This viewer requires WebGL 2.';return;}
  function program(v,f){
    const p=gl.createProgram();for(const [type,src] of [[gl.VERTEX_SHADER,v],[gl.FRAGMENT_SHADER,f]]){
      const sh=gl.createShader(type);gl.shaderSource(sh,src);gl.compileShader(sh);
      if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(sh));gl.attachShader(p,sh);
    }
    gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));return p;
  }
  const screen=program(`#version 300 es
  out vec2 uv;void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);uv=p;gl_Position=vec4(2.*p-1.,0,1);}`,`#version 300 es
  precision highp float;in vec2 uv;uniform sampler2D img;uniform bool bare;uniform vec2 origin,right,up;out vec4 col;
  void main(){vec2 p=origin+right*(2.*uv.x-1.)+up*(2.*uv.y-1.);col=bare?vec4(all(lessThanEqual(abs(p),vec2(5000.)))?vec3(80,72,81)/255.:vec3(28,30,36)/255.,1):texture(img,uv);}`);
  const uniforms=Object.fromEntries(['bare','origin','right','up'].map(k=>[k,gl.getUniformLocation(screen,k)]));
  const reference=makeStrandReference(gl,program);
  gl.disable(gl.DITHER);
  const tex=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,tex);
  for(const k of [gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_2D,k,gl.LINEAR);
  for(const k of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T])gl.texParameteri(gl.TEXTURE_2D,k,gl.CLAMP_TO_EDGE);
  let texW=0,texH=0,pixels=new Uint32Array(0);
  let az=-1.15,el=.87,zoom=8,x=0,y=0,stroke=null,tool='orbit',drag=null,walking=false,busy=false,raf=0,previous=0,report;
  const preset=()=>Field.stroke(0,-5.5,0,5.5),cam=()=>Field.camera(az,el,zoom,x,y);
  function quad(m,bytes,w,h){
    gl.useProgram(screen);gl.uniform2f(uniforms.origin,m.ox,m.oy);gl.uniform2f(uniforms.right,m.rx,m.ry);gl.uniform2f(uniforms.up,m.ux,m.uy);gl.uniform1i(uniforms.bare,bytes?0:1);
    if(bytes){if(texW!==w||texH!==h){gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,w,h,0,gl.RGBA,gl.UNSIGNED_BYTE,bytes);texW=w;texH=h;}
      else gl.texSubImage2D(gl.TEXTURE_2D,0,0,0,w,h,gl.RGBA,gl.UNSIGNED_BYTE,bytes);}
    gl.drawArrays(gl.TRIANGLES,0,3);
  }
  const patchProgram=program(`#version 300 es
  precision highp float;uniform vec4 box,axes,address;uniform vec2 origin;out vec2 uv;
  void main(){int id=gl_VertexID;vec2 f=vec2(id==1||id==4||id==5?1.:0.,id==2||id==3||id==5?1.:0.);
    vec2 p=mix(box.xy,box.zw,f);uv=(p-address.xy)*address.zw;
    gl_Position=vec4(dot(p-origin,axes.xy),dot(p-origin,axes.zw),0,1);}
  `,`#version 300 es
  precision highp float;in vec2 uv;uniform sampler2D img;out vec4 color;void main(){color=texture(img,uv);}`);
  const pu=Object.fromEntries(['box','axes','address','origin'].map(k=>[k,gl.getUniformLocation(patchProgram,k)]));
  const textures=[];
  function drawField(m,b,w,h,n){
    const detail=+$('quality').value;
    if(!detail){
      if(pixels.length!==w*h)pixels=new Uint32Array(w*h);
      const count=Field.renderNative(m,b,w,h,n,pixels),cpu=performance.now();
      quad(m,new Uint8Array(pixels.buffer),w,h);return {count,cpu};
    }
    gl.useProgram(patchProgram);
    const rx2=m.rx*m.rx+m.ry*m.ry,uy2=m.ux*m.ux+m.uy*m.uy;
    gl.uniform4f(pu.axes,m.rx/rx2,m.ry/rx2,m.ux/uy2,m.uy/uy2);gl.uniform2f(pu.origin,m.ox,m.oy);
    const sent=new Map();let uploadTime=0;
    const start=performance.now();
    const result=Field.patches(m,b,detail,n,(entry,l,r,d,u,x0,y0,tw,th,wrapX,wrapY)=>{
      const now=performance.now();let texture=sent.get(entry.key);
      if(!texture){
        const i=sent.size;texture=textures[i]||(textures[i]=gl.createTexture());gl.bindTexture(gl.TEXTURE_2D,texture);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
        gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,tw,th,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array(entry.data.buffer));
        sent.set(entry.key,texture);
      }else gl.bindTexture(gl.TEXTURE_2D,texture);
      gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,wrapX?gl.REPEAT:gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,wrapY?gl.REPEAT:gl.CLAMP_TO_EDGE);
      gl.uniform4f(pu.box,l,d,r,u);gl.uniform4f(pu.address,x0,y0,detail/tw,detail/th);
      gl.drawArrays(gl.TRIANGLES,0,6);uploadTime+=performance.now()-now;
    });
    gl.bindTexture(gl.TEXTURE_2D,tex);
    return {count:result.samples,cpu:start+(performance.now()-start-uploadTime)};
  }
  function drawRef(m,b,levels=40,sectors=12){quad(m);return reference(m,b,levels,sectors);}
  function read(){const data=new Uint8Array(canvas.width*canvas.height*4);gl.readPixels(0,0,canvas.width,canvas.height,gl.RGBA,gl.UNSIGNED_BYTE,data);return data;}
  function errors(a,b,diff=false){
    let sum=0,large=0,skinMis=0;for(let i=0;i<a.length;i+=4){
      const da=Math.abs(a[i]-b[i]),db=Math.abs(a[i+1]-b[i+1]),dc=Math.abs(a[i+2]-b[i+2]),d=Math.max(da,db,dc);
      sum+=da+db+dc;large+=d>16;
      skinMis+=((a[i]===80&&a[i+1]===72&&a[i+2]===81)!==(b[i]===80&&b[i+1]===72&&b[i+2]===81));
      if(diff){a[i]=Math.min(255,d*4);a[i+1]=Math.min(255,d);a[i+2]=Math.min(255,d*3);}
    }return {mae:sum/(a.length*.75),over16:large*400/a.length,backingDisagreement:skinMis*400/a.length};
  }
  function dims(){return [canvas.width,canvas.height];}
  function resize(w,h){if(w!==canvas.width||h!==canvas.height){canvas.width=w;canvas.height=h;}gl.viewport(0,0,w,h);}
  function refresh(){
    if(busy||raf)return;
    raf=requestAnimationFrame(t=>{raf=0;if(busy)return;
      if(walking){az+=Math.min(150,t-previous)*.0005;previous=t;}
      let w=Math.min(1440,Math.max(320,Math.round(canvas.getBoundingClientRect().width*devicePixelRatio)));resize(w,Math.round(w*9/16));
      const m=cam(),mode=$('mode').value,start=performance.now();let sampled=0;
      if(mode==='field')sampled=drawField(m,stroke,...dims(),32).count;
      if(mode==='strands')drawRef(m,stroke);
      if(mode==='bare')quad(m);
      if(mode==='diff'){drawField(m,stroke,...dims(),32);const a=read();drawRef(m,stroke);errors(a,read(),true);quad(m,a,canvas.width,canvas.height);}
      status.textContent=(stroke?'1 brush field':'Unbrushed')+' · '+Math.round(el*180/Math.PI)+'° elevation'+(sampled?' · '+sampled.toLocaleString()+' field samples':'')+' · '+Math.round(performance.now()-start)+' ms to submit this view'+(report?'\nLast 4K orbit: '+report.summary:'');
      if(walking)refresh();
    });
  }
  function setTool(t){tool=t;for(const k of ['orbit','brush'])$(k).setAttribute('aria-pressed',String(k===t));canvas.classList.toggle('brushing',t==='brush');}
  $('orbit').onclick=()=>setTool('orbit');$('brush').onclick=()=>setTool('brush');
  $('preset').onclick=()=>{stroke=preset();refresh();};$('clear').onclick=()=>{stroke=null;refresh();};
  $('home').onclick=()=>{az=-1.15;el=.87;zoom=8;x=y=0;refresh();};
  function ground(e){const r=canvas.getBoundingClientRect(),sx=2*(e.clientX-r.left)/r.width-1,sy=1-2*(e.clientY-r.top)/r.height,m=cam();
    return [m.ox+m.rx*sx+m.ux*sy,m.oy+m.ry*sx+m.uy*sy];}
  canvas.onpointerdown=e=>{if(busy)return;canvas.setPointerCapture(e.pointerId);drag={start:ground(e),x:e.clientX,y:e.clientY,pan:e.shiftKey};};
  canvas.onpointermove=e=>{if(!drag||busy)return;
    const p=ground(e);if(drag.pan){x+=drag.start[0]-p[0];y+=drag.start[1]-p[1];}
    else if(tool==='brush')stroke=Field.stroke(...drag.start,...p);
    else{az-=(e.clientX-drag.x)*.009;el=Math.max(.25,Math.min(1.54,el+(e.clientY-drag.y)*.008));}
    drag.x=e.clientX;drag.y=e.clientY;refresh();};
  canvas.onpointerup=canvas.onpointercancel=()=>{drag=null;};
  canvas.onwheel=e=>{e.preventDefault();if(!busy){zoom=Math.min(25,Math.max(2,zoom*Math.exp(e.deltaY*.001)));refresh();}};
  $('walk').onclick=()=>{walking=!walking;previous=performance.now();$('walk').textContent=walking?'Stop walk':'Walk around';refresh();};
  $('quality').onchange=$('mode').onchange=refresh;
  const tick=()=>new Promise(r=>requestAnimationFrame(r)),sync=new Uint8Array(3840*2160*4);
  const finish=()=>gl.readPixels(0,0,3840,2160,gl.RGBA,gl.UNSIGNED_BYTE,sync);
  function median(a){a=[...a].sort((a,b)=>a-b);return .5*(a[(a.length-1)>>1]+a[a.length>>1]);}
  async function benchmark(count=12){
    if(busy)return;busy=true;walking=false;$('walk').textContent='Walk around';drag=null;
    for(const e of document.querySelectorAll('button,select'))e.disabled=true;
    try{
      resize(3840,2160);if(gl.drawingBufferWidth!==3840||gl.drawingBufferHeight!==2160)throw Error('4K buffer allocation failed.');
      const [w,h]=dims(),rows=[],savedStroke=stroke;
      const bare=m=>{const t=performance.now();quad(m);finish();return performance.now()-t;};
      const model=m=>{const t=performance.now(),r=drawField(m,savedStroke,w,h,32);finish();return {full:performance.now()-t,cpu:r.cpu-t,direct:r.count};};
      const first=cam();bare(first);model(first);drawRef(first,savedStroke);finish();
      for(let i=0;i<count;i++){
        status.textContent='Measuring 4K view '+(i+1)+' / '+count;await tick();
        const theta=i*2*Math.PI/count,m=Field.camera(-1.15+theta,.82+.34*Math.sin(theta),zoom*(.93+.15*Math.cos(theta)),x,y);
        let base,field,a;if(i%2){field=model(m);a=sync.slice();base=bare(m);}else{base=bare(m);field=model(m);a=sync.slice();}
        const start=performance.now(),roots=drawRef(m,savedStroke);finish();const refMs=performance.now()-start,ref=sync.slice();
        const row={pose:i,az:m.az,elevation:m.el,zoom:m.zoom,bare:base,full:field.full,cpu:field.cpu,added:field.full-base,fieldEvaluations:field.direct,referenceMs:refMs,explicitRootCandidates:roots,...errors(a,ref)};
        if(i===0||i===Math.floor(count/2)){
          drawRef(m,savedStroke,80,24);finish();row.referenceRefinement=errors(ref,sync);
          // Split display sampling error from sampling the volume at finite heights.
          drawField(m,savedStroke,w,h,64);row.doubleDepthVsReference=errors(read(),ref);
        }
        rows.push(row);
      }
      const med=k=>median(rows.map(r=>r[k])),summary=med('full').toFixed(2)+' ms full · '+med('added').toFixed(2)+' ms fur added';
      report={frame:[3840,2160],samplesPerPitch:+$('quality').value||null,nativePixelSolve:+$('quality').value===0,heightBands:32,impliedHairs:100000000,brush:savedStroke,synchronization:"full 4K RGBA readback included in paired times",rows,summary};return report;
    }finally{busy=false;for(const e of document.querySelectorAll('button,select'))e.disabled=false;$('save').disabled=!report;refresh();}
  }
  $('bench').onclick=()=>benchmark().catch(e=>status.textContent=e.message);
  $('save').onclick=()=>{const a=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));a.href=url;a.download='curl-field-4k.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  window.fieldStudy={benchmark,pose(a,e,z){az=a;el=e;zoom=z;refresh();},setBrush(b){stroke=b;refresh();},stroke:Field.stroke,cam};
  new ResizeObserver(refresh).observe(canvas);refresh();
})()
