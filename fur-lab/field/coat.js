'use strict';
// All distances are in root pitches. Each half has 5,000 x 10,000 periods.
const Field = (() => {
  const half=5000, height=3.8, tau=2*Math.PI, skin=0xff514850, air=0xff241e1c;
  const pigments=[[177,107,61],[149,171,184]];
  function curl(side,t) {
    const a=side?.47:.69, phase=side?2.1:.35, omega=(side?-1.62:1.03)*tau;
    const angle=phase+omega*t, lean=side?.28:-.34;
    return [a*(Math.cos(angle)-Math.cos(phase))+lean*t,a*(Math.sin(angle)-Math.sin(phase))+.18*t,
      -a*omega*Math.sin(angle)+lean,a*omega*Math.cos(angle)+.18];
  }
  function stroke(ax,ay,bx,by){
    const dx=bx-ax,dy=by-ay,len=Math.hypot(dx,dy);
    if(len<.03)return null;
    return {x:(ax+bx)/2,y:(ay+by)/2,sx:dx/len,sy:dy/len,nx:-dy/len,ny:dx/len,L:len/2};
  }
  // In stroke coordinates, q'=q+A*g(q)*f(s)*t², s'=s.
  // g is odd, piecewise linear; outside width 5 it vanishes. The outer
  // slope remains positive even at the tips: 1-2.65/(5-.8) > 0.
  const inner=.8, width=5, amp=2.65, feather=2;
  function shade(side,t,dx,dy,light){
    const inv=1/Math.sqrt(dx*dx+dy*dy+height*height);
    const dot=(-.5*dx-.4*dy+height*Math.sqrt(.59))*inv,h=(dx*light[0]+dy*light[1]+height*light[2])*inv;
    let shine=Math.max(0,1-h*h);shine*=shine;shine*=shine;shine*=shine;
    const lum=(.34+.40*Math.sqrt(Math.max(0,1-dot*dot))+.43*shine)*(.73+.27*t);
    const rgb=pigments[side];
    return (0xff000000|(Math.min(255,Math.round(rgb[2]*lum))<<16)|(Math.min(255,Math.round(rgb[1]*lum))<<8)|Math.min(255,Math.round(rgb[0]*lum)))>>>0;
  }
  function camera(az,el,zoom,x=0,y=0){
    const c=Math.cos(az),s=Math.sin(az),q=1/Math.tan(el),k=zoom*9/16/Math.sin(el);
    return {az,el,zoom,x,y,ox:x-c*q*height*.52,oy:y-s*q*height*.52,rx:-s*zoom,ry:c*zoom,ux:-c*k,uy:-s*k,vx:c*q*height,vy:s*q*height};
  }
  function backing(x,y){return Math.abs(x)<=half&&Math.abs(y)<=half?skin:air;}
  function layers(m,n){
    const hx=-.5+Math.cos(m.az)*Math.cos(m.el),hy=-.4+Math.sin(m.az)*Math.cos(m.el),hz=Math.sqrt(.59)+Math.sin(m.el);
    const length=Math.hypot(hx,hy,hz),light=[hx/length,hy/length,hz/length],out=[];
    for(let k=0;k<=n;k++){
      const t=1-k/n, r=.168*(1-.57*t),curves=[curl(0,t),curl(1,t)];
      out.push({t,light,curves,r,r2:r*r,bend:amp*t*t,vx:m.vx*t,vy:m.vy*t,
        ax:curves[0][0],ay:curves[0][1],bx:curves[1][0],by:curves[1][1],
        ca:shade(0,t,curves[0][2],curves[0][3],light),
        cb:shade(1,t,curves[1][2],curves[1][3],light)});
    }
    return out;
  }
  function bentColor(side,h,b,u,v){
    const dx=u-b.x,dy=v-b.y,s=dx*b.sx+dy*b.sy,q=dx*b.nx+dy*b.ny,d=Math.abs(q);
    if(d>=width||Math.abs(s)>=b.L+feather)return side?h.cb:h.ca;
    const f=Math.min(1,(b.L+feather-Math.abs(s))/feather),df=Math.abs(s)>b.L?-Math.sign(s)/feather:0;
    const g=Math.sign(q)*(d<inner?d/inner:(width-d)/(width-inner)),dg=d<inner?1/inner:-1/(width-inner);
    const vx=h.curves[side][2],vy=h.curves[side][3];
    const dv=amp*(2*h.t*g*f+h.t*h.t*(dg*(vx*b.nx+vy*b.ny)*f+g*df*(vx*b.sx+vy*b.sy)));
    return shade(side,h.t,vx+b.nx*dv,vy+b.ny*dv,h.light);
  }
  // For each subinterval of one period, retain a bit for heights where its
  // projection could intersect the fiber disk. The bound includes the entire
  // phase bin, both row staggers and the largest possible inverse brush shift.
  // AND of x/y masks excludes only provably empty sampled heights.
  const bins=256;
  function heightMasks(ls,b){
    const words=Math.ceil((ls.length-1)/32),tables=Array.from({length:4},()=>new Uint32Array(words*bins));
    for(let side=0;side<2;side++)for(let axis=0;axis<2;axis++){
      const table=tables[2*side+axis],normal=b?Math.abs(axis?b.ny:b.nx):0;
      for(let k=0;k<ls.length-1;k++){
        const h=ls[k],lo=ls[k+1],offset=(axis?h.vy:h.vx)-(side?(axis?h.by:h.bx):(axis?h.ay:h.ax));
        const lowerOffset=(axis?lo.vy:lo.vx)-(side?(axis?lo.by:lo.bx):(axis?lo.ay:lo.ax));
        const bound=Math.sqrt(lo.r2)+.5/bins+h.bend*normal+Math.abs(lowerOffset-offset)*.5+1e-9;
        for(let p=0;p<bins;p++){
          let v=(p+.5)/bins+(offset+lowerOffset)*.5;
          const d=axis?Math.abs(v-Math.floor(v)-.5):
            Math.min(Math.abs(v+.16-Math.floor(v+.16)-.5),Math.abs(v-.16-Math.floor(v-.16)-.5));
          if(d<=bound)table[p*words+(k>>>5)]|=1<<(k&31);
        }
      }
    }
    return {words,tables};
  }
  function unbend(u,v,h,b){
    if(!b)return 0;
    const dx=u-b.x,dy=v-b.y,s=dx*b.sx+dy*b.sy,q=dx*b.nx+dy*b.ny,d=Math.abs(q);
    if(d>=width||Math.abs(s)>=b.L+feather)return 0;
    const a=h.bend*Math.min(1,(b.L+feather-Math.abs(s))/feather);
    const inv=d<=inner+a?d*inner/(inner+a):(d-a*width/(width-inner))/(1-a/(width-inner));
    return Math.sign(q)*(inv-d);
  }
  // Intersect the phase path through one height band with the tapered disk
  // equation. Within a short band only the period containing its midpoint can
  // hit. Split long phase paths (grazing rays or a strong part) geometrically.
  function crossing(x,y,dx,dy,r,dr,side){
    if(Math.max(Math.abs(dx),Math.abs(dy))>.64){
      let a=crossing(x,y,dx*.5,dy*.5,r,dr*.5,side);if(a<=1)return a*.5;
      let b=crossing(x+dx*.5,y+dy*.5,dx*.5,dy*.5,r+dr*.5,dr*.5,side);return b<=1?.5+b*.5:2;
    }
    const mx=x+dx*.5,my=y+dy*.5,j=Math.floor(my);
    const px=mx-((j&1)?.16:-.16),ix=Math.floor(px);
    if(j < -half||j>=half||ix < (side?0:-half)||ix >= (side?half:0))return 2;
    const fx=px-ix-.5-dx*.5,fy=my-j-.5-dy*.5,C=fx*fx+fy*fy-r*r;
    if(C<=0)return 0;
    const A=dx*dx+dy*dy-dr*dr,B=fx*dx+fy*dy-r*dr,D=B*B-A*C;
    if(D<0)return 2;
    // Stable smaller positive root, including a nearly linear inequality.
    const den=Math.sqrt(D)-B;if(den<=0||C>den)return 2;return C/den;
  }
  function resolve(x,y,ls,b,masks){
    const {words,tables}=masks,ix=Math.floor((x-Math.floor(x))*bins)*words,iy=Math.floor((y-Math.floor(y))*bins)*words;
    const nx=b?b.nx:0,ny=b?b.ny:0;
    for(let w=0;w<words;w++){
      const left=tables[0][ix+w]&tables[1][iy+w],right=tables[2][ix+w]&tables[3][iy+w];
      let bits=left|right;
      while(bits){
        const bit=bits&-bits,k=32*w+31-Math.clz32(bit);bits&=bits-1;
        const h=ls[k],l=ls[k+1],u=x+h.vx,v=y+h.vy,u1=x+l.vx,v1=y+l.vy;
        const d=unbend(u,v,h,b),d1=unbend(u1,v1,l,b),X=u+d*nx,Y=v+d*ny;
        const dx=(u1-u)+(d1-d)*nx,dy=(v1-v)+(d1-d)*ny,r=h.r,dr=l.r-r;
        const a=left&bit?crossing(X-h.ax,Y-h.ay,dx+h.ax-l.ax,dy+h.ay-l.ay,r,dr,0):2;
        const c=right&bit?crossing(X-h.bx,Y-h.by,dx+h.bx-l.bx,dy+h.by-l.by,r,dr,1):2;
        const t=Math.min(a,c);
        if(t<=1){
          const side=Number(c<a);
          const h0=b?bentColor(side,h,b,X,Y):(side?h.cb:h.ca);
          const h1=b?bentColor(side,l,b,X+dx,Y+dy):(side?l.cb:l.ca);
          const R=(h0&255)+t*((h1&255)-(h0&255)),G=((h0>>>8)&255)+t*(((h1>>>8)&255)-((h0>>>8)&255));
          const B=((h0>>>16)&255)+t*(((h1>>>16)&255)-((h0>>>16)&255));
          return (0xff000000|(Math.round(B)<<16)|(Math.round(G)<<8)|Math.round(R))>>>0;
        }
      }
    }
    return backing(x,y);
  }
  function renderNative(m,b,w,h,n,pixels){
    const ls=layers(m,n),plain=heightMasks(ls,null),warped=b?heightMasks(ls,b):plain;
    for(let j=0;j<h;j++){
      const sy=(j+.5)*2/h-1,dx=2*m.rx/w,dy=2*m.ry/w;
      let x=m.ox+sy*m.ux-m.rx+dx*.5,y=m.oy+sy*m.uy-m.ry+dy*.5;
      for(let i=0;i<w;i++,x+=dx,y+=dy){
        let active=false;
        if(b){
          const xx=x-b.x,yy=y-b.y,along=xx*b.sx+yy*b.sy,normal=xx*b.nx+yy*b.ny;
          const ds=m.vx*b.sx+m.vy*b.sy,dq=m.vx*b.nx+m.vy*b.ny;
          active=Math.abs(normal+dq*.5)<=width+Math.abs(dq)*.5&&Math.abs(along+ds*.5)<=b.L+feather+Math.abs(ds)*.5;
        }
        pixels[j*w+i]=resolve(x,y,ls,active?b:null,active?warped:plain);
      }
    }
    return w*h;
  }
  // Subdivide the finished backing-plane picture only at boundaries of known
  // translation symmetries. A patch repeats an axis only if translating it by
  // one lattice period preserves both fields and the brush along the entire ray.
  function patches(m,b,res,n,paint){
    const ls=layers(m,n),plain=heightMasks(ls,null),warped=b?heightMasks(ls,b):plain;
    const ex=Math.abs(m.rx)+Math.abs(m.ux)+.01,ey=Math.abs(m.ry)+Math.abs(m.uy)+.01;
    const guard=2/res,loX=m.ox-ex,hiX=m.ox+ex,loY=m.oy-ey,hiY=m.oy+ey;
    const xcuts=[loX,hiX],ycuts=[loY,hiY];
    const cut=(arr,l,h,v)=>{if(l<v&&v<h)arr.push(v);};
    // All curl-only material points lie within these offsets of the backing hit.
    const xPad=Math.abs(m.vx)+2.1+guard,yPad=Math.abs(m.vy)+1.9+guard;
    for(const v of [-half+xPad,-xPad,xPad,half-xPad])cut(xcuts,loX,hiX,v);
    for(const v of [-half+yPad,half-yPad])cut(ycuts,loY,hiY,v);
    let xmin=0,xmax=0,ymin=0,ymax=0,flatLo=0,flatHi=0;
    const vertical=b&&b.sx===0,horizontal=b&&b.sy===0;
    if(b){
      const dx=Math.abs(b.nx)*width+Math.abs(b.sx)*(b.L+feather),dy=Math.abs(b.ny)*width+Math.abs(b.sy)*(b.L+feather);
      xmin=b.x-dx-Math.max(0,m.vx)-guard;xmax=b.x+dx-Math.min(0,m.vx)+guard;
      ymin=b.y-dy-Math.max(0,m.vy)-guard;ymax=b.y+dy-Math.min(0,m.vy)+guard;
      for(const v of [xmin,xmax])cut(xcuts,loX,hiX,v);for(const v of [ymin,ymax])cut(ycuts,loY,hiY,v);
      if(vertical||horizontal){
        const mid=vertical?b.y:b.x,v=vertical?m.vy:m.vx;
        flatLo=mid-b.L-Math.min(0,v)+guard;flatHi=mid+b.L-Math.max(0,v)-guard;
        if(flatLo<flatHi)for(const k of [flatLo,flatHi])vertical?cut(ycuts,loY,hiY,k):cut(xcuts,loX,hiX,k);
      }
    }
    xcuts.sort((a,b)=>a-b);ycuts.sort((a,b)=>a-b);
    let samples=0;const cached=new Map();
    for(let j=1;j<ycuts.length;j++)for(let i=1;i<xcuts.length;i++){
      const l=xcuts[i-1],r=xcuts[i],d=ycuts[j-1],u=ycuts[j],x=(l+r)*.5,y=(d+u)*.5;
      if(r<=l||u<=d)continue;
      const touched=b&&x>xmin&&x<xmax&&y>ymin&&y<ymax;
      const xSafe=Math.abs(x)>xPad&&Math.abs(x)<half-xPad,ySafe=Math.abs(y)<half-yPad;
      // Brush displaces along its normal; a horizontal stroke has no x displacement.
      const repeatX=xSafe&&(!touched||(horizontal&&x>flatLo&&x<flatHi));
      const repeatY=ySafe&&(!touched||(vertical&&y>flatLo&&y<flatHi));
      // A folded strip is evaluated at an integer translate of this very patch.
      // For a plateau the longitudinal brush weight is exactly one there; force
      // that constant extension so a canonical period can also cover a narrow strip.
      let effective=touched?b:null;
      if(touched&&((vertical&&repeatY)||(horizontal&&repeatX)))effective={...b,L:Infinity};
      // Pad nonrepeating sides for consistent bilinear filtering across patches.
      const x0=repeatX?(x>0?100:-100):Math.floor(l*res-1)/res;
      const y0=repeatY?0:Math.floor(d*res-1)/res;
      const w=repeatX?res:Math.ceil(r*res+1)-Math.round(x0*res);
      const h=repeatY?2*res:Math.ceil(u*res+1)-Math.round(y0*res);
      const key=[x0,y0,w,h,!!effective,effective?.L].join('/');
      let entry=cached.get(key);
      if(!entry){
        const data=new Uint32Array(w*h),mask=effective?warped:plain;
        for(let v=0;v<h;v++)for(let q=0;q<w;q++)data[v*w+q]=resolve(x0+(q+.5)/res,y0+(v+.5)/res,ls,effective,mask);
        entry={data,key};cached.set(key,entry);samples+=data.length;
      }
      paint(entry,l,r,d,u,x0,y0,w,h,repeatX,repeatY);
    }
    return {samples};
  }
  return {camera,stroke,renderNative,patches};
})();
