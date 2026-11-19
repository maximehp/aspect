'use strict';
// Coordinates are in root spacings. The image lives on the backing plane.
const Coat = (() => {
  const extent=20, bands=12, radius=.20;
  const tau=Math.PI*2;
  // Finite 10,000 x 10,000 material lattice. IDs 0 and 99,999,999 are its opposite corners.
  const half=5000, width=2*half, edits=new Map(),editedColumns=new Map();
  let originX=0, originY=0;
  const address=(i,j)=>(j+half)*width+i+half;
  const rowOffset=(j,side)=>.42*Math.sin(79.1*j+side*1.7);
  const rowY=j=>j+.15*Math.sin(j*12.7);
  function locate(x,y){
    originX=Math.max(-half,Math.min(half-1,Math.round(x)));
    originY=Math.max(-half,Math.min(half-1,Math.round(y)));
  }
  function location(){return [originX,originY];}
  function reset(){edits.clear();editedColumns.clear();}
  function brush(ax,ay,bx,by){
    ax+=originX;bx+=originX;ay+=originY;by+=originY;
    const dx=bx-ax,dy=by-ay,len=Math.hypot(dx,dy);if(len<.01)return 0;
    const nx=-dy/len,ny=dx/len;
    let count=0;
    for(let j=Math.max(-half,Math.floor(Math.min(ay,by)-3));j<=Math.min(half-1,Math.ceil(Math.max(ay,by)+3));j++){
      for(let i=Math.max(-half,Math.floor(Math.min(ax,bx)-4));i<=Math.min(half-1,Math.ceil(Math.max(ax,bx)+4));i++){
        let x=i+rowOffset(j,i<0?-1:1)-ax,y=rowY(j)-ay;
        let t=Math.max(0,Math.min(1,(x*dx+y*dy)/(len*len)));
        let dist=Math.hypot(x-t*dx,y-t*dy);if(dist>=2.85)continue;
        let sign=x*nx+y*ny>=0?1:-1,angle=Math.atan2(sign*ny,sign*nx);
        let dir=(Math.round(angle/tau*16)+16)%16;
        let state=1+dir+16*Math.min(3,Math.floor(4*(1-dist/2.85))),id=address(i,j);
        if(edits.get(id)!==state){
          if(!edits.has(id)){
            const columns=editedColumns.get(j)||[];let pos=columns.findIndex(col=>col>i);
            if(pos<0)columns.push(i);else columns.splice(pos,0,i);
            editedColumns.set(j,columns);
          }
          edits.set(id,state);count++;
        }
      }
    }
    return count;
  }
  function part(){return brush(0,-6.8,0,6.8);}
  function edited(){return edits.size;}
  function prepare(az,el,extra=0){
    const skewX=Math.cos(az)/Math.tan(el),skewY=Math.sin(az)/Math.tan(el),runs=[];
    // Independent coordinate bounds: row jitter, twice the curl radius, guide lean,
    // brush bend (<= 2.80), disk radius and height (<= 4.01) under this projection.
    const rangeX=Math.ceil(extent+5.4+4.01*Math.abs(skewX))+extra;
    const rangeY=Math.ceil(extent+5.4+4.01*Math.abs(skewY))+extra;
    for(let j=Math.max(-half,originY-rangeY);j<=Math.min(half-1,originY+rangeY);j++){
      const end=Math.min(half-1,originX+rangeX),columns=editedColumns.get(j)||[];
      let cursor=0;
      for(let i=Math.max(-half,originX-rangeX);i<=end;){
        const side=i<0?-1:1, cell=Math.floor(i/16);
        const state=edits.get(address(i,j))||0,limit=Math.min(end,(cell+1)*16-1);
        while(cursor<columns.length&&columns[cursor]<i)cursor++;
        let stop=state?i:Math.min(limit,(columns[cursor]??Infinity)-1);
        while(state&&stop<limit&&edits.get(address(stop+1,j))===state)stop++;
        const first=i-originX,last=stop-originX;i=stop+1;
        const row={side,y:rowY(j)-originY,offset:rowOffset(j,side)};
        const phase=j*.69+side*.8+.85*Math.sin(cell*2.07+j*.29);
        const turn=side<0?1.08:1.72,amp=side<0?.68:.48;
        let bx=0,by=0;if(state){let ang=(state-1)%16*tau/16,m=.85+.65*Math.floor((state-1)/16);bx=m*Math.cos(ang);by=m*Math.sin(ang);}
        let height=(3.7+.31*Math.cos(phase))*(1-.11*Math.hypot(bx,by));
        const segs=[];let prev;
        for(let k=0;k<=bands;k++){
          const t=k/bands,angle=phase+row.side*turn*tau*t;
          let x=amp*(Math.cos(angle)-Math.cos(phase))+.35*row.side*t+bx*t*t;
          let y=amp*(Math.sin(angle)-Math.sin(phase))+.26*t+by*t*t,z=height*t;
          let point={x,y,z,u:row.offset+x-skewX*z,v:row.y+y-skewY*z};
          if(k){
            const dx=x-prev.x,dy=y-prev.y,dz=z-prev.z,n=Math.hypot(dx,dy,dz);
            const lx=-.50,ly=-.40,lz=Math.sqrt(.59);
            const hx=lx+Math.cos(az)*Math.cos(el),hy=ly+Math.sin(az)*Math.cos(el),hz=lz+Math.sin(el);
            const h=(dx*hx+dy*hy+dz*hz)/(n*Math.hypot(hx,hy,hz));
            const diffuse=Math.sqrt(Math.max(0,1-((dx*lx+dy*ly+dz*lz)/n)**2));
            const shine=.39+.40*diffuse+.38*(Math.max(0,1-h*h)**8);
            const pigment=row.side<0?[181,126,78]:[139,157,169];
            const col=pigment.map(a=>Math.min(255,Math.round(a*shine*(.73+.27*t))));
            segs.push({u:prev.u,v:prev.v,du:point.u-prev.u,dv:point.v-prev.v,z:prev.z,dz,r:radius*(1-.67*(k-.5)/bands),color:(0xff000000|col[2]<<16|col[1]<<8|col[0])>>>0});
          }
          prev=point;
        }
        runs.push({first,last,segs,firstID:address(first+originX,j)});
      }
    }
    return runs;
  }
  const bg=0xff25221e,skin=0xff625455;
  function base(x,y){return x+originX>=-half-.5&&x+originX<half-.5&&y+originY>=-half-.5&&y+originY<half-.5?skin:bg;}
  // Highest point of intersection with a single finite swept horizontal disk.
  function hit(s,x,y){
    let a=s.du*s.du+s.dv*s.dv,b=x*s.du+y*s.dv,c=x*x+y*y-s.r*s.r;
    if(a<1e-18)return c<=0?s.z+s.dz:-Infinity;
    let disc=b*b-a*c;if(disc<0)return -Infinity;let d=Math.sqrt(disc);
    let hi=Math.min(1,(b+d)/a);return hi>=Math.max(0,(b-d)/a)?s.z+s.dz*hi:-Infinity;
  }
  function makeAtlas(res){
    let n=extent*2*res,b=new Uint32Array(n*n);
    for(let j=0;j<n;j++)for(let i=0;i<n;i++)b[j*n+i]=base((i+.5)/res-extent,(j+.5)/res-extent);
    return {n,res,base:b,pixels:b.slice(),depth:new Float64Array(n*n)};
  }
  function render(runs,a,reference){
    const {n,res,pixels,depth}=a;pixels.set(a.base);depth.fill(-Infinity);
    const idx=x=>Math.max(0,Math.min(n,Math.ceil((x+extent)*res-.5)));
    const end=x=>Math.max(0,Math.min(n,Math.floor((x+extent)*res-.5)+1));
    const pattern=new Float64Array(res);
    for(const g of runs)for(const s of g.segs){
      let left=Math.min(s.u,s.u+s.du)-s.r,right=Math.max(s.u,s.u+s.du)+s.r;
      let bottom=idx(Math.min(s.v,s.v+s.dv)-s.r),top=end(Math.max(s.v,s.v+s.dv)+s.r);
      if(reference){
        // Explicitly project and intersect every individual fiber piece.
        for(let root=g.first;root<=g.last;root++)for(let j=bottom;j<top;j++){
          let y=(j+.5)/res-extent-s.v,from=idx(root+left),to=end(root+right);
          for(let i=from;i<to;i++){
            let z=hit(s,(i+.5)/res-extent-root-s.u,y),p=j*n+i;
            if(z>depth[p]){depth[p]=z;pixels[p]=s.color;}
          }
        }
      }else{
        // Response of one period of a coherent run, with exact finite end treatment.
        // Integer root translations shift atlas columns by exactly res samples.
        for(let j=bottom;j<top;j++){
          let y=(j+.5)/res-extent-s.v;
          const from=idx(g.first+left),to=end(g.last+right);
          for(let ph=0;ph<res;ph++){
            const x=(ph+.5)/res;let z=-Infinity;
            for(let r=Math.ceil(x-right);r<=Math.floor(x-left);r++)z=Math.max(z,hit(s,x-r-s.u,y));
            pattern[ph]=z;
          }
          const lo=idx(g.first-1+right)+1,hi=idx(g.last+1+left)-1;
          for(let i=from;i<to;i++){
            let z=pattern[i%res];if(z===-Infinity)continue;
            if(i<lo||i>=hi){
              let x=(i+.5)/res-extent;z=-Infinity;
              for(let r=Math.max(g.first,Math.ceil(x-right));r<=Math.min(g.last,Math.floor(x-left));r++)z=Math.max(z,hit(s,x-r-s.u,y));
            }
            const p=j*n+i;if(z>depth[p]){depth[p]=z;pixels[p]=s.color;}
          }
        }
      }
    }
    return a;
  }
  // Independent point reference at arbitrary backing-plane coordinates for the final-display sampling check.
  function sample(runs,x,y,contact=false){
    let z=-Infinity,col=base(x,y),rootID=null;
    for(const g of runs)for(const s of g.segs){
      if(y<Math.min(s.v,s.v+s.dv)-s.r||y>Math.max(s.v,s.v+s.dv)+s.r)continue;
      let l=Math.max(g.first,Math.ceil(x-Math.max(s.u,s.u+s.du)-s.r)),h=Math.min(g.last,Math.floor(x-Math.min(s.u,s.u+s.du)+s.r));
      for(let r=l;r<=h;r++){let zz=hit(s,x-r-s.u,y-s.v);if(zz>z){z=zz;col=s.color;rootID=g.firstID+r-g.first;}}
    }
    return contact?(z===-Infinity?null:{z,rootID}):col;
  }
  // Native-pixel reference. Project each individual segment's bounding box, then
  // intersect the actual 4K pixel-center sightline inside it.
  function native(runs,m,w,h,target){
    const n=w*h;
    if(!target.pixels||target.pixels.length!==n){target.pixels=new Uint32Array(n);target.depth=new Float64Array(n);}
    const pixels=target.pixels,depth=target.depth;depth.fill(-Infinity);
    const xx=m[1][0]*2/w,xy=m[1][1]*2/w,yx=m[2][0]*2/h,yy=m[2][1]*2/h;
    const ox=m[0][0]-m[1][0]-m[2][0]+.5*(xx+yx),oy=m[0][1]-m[1][1]-m[2][1]+.5*(xy+yy);
    const det=xx*yy-xy*yx,ix=yy/det,iy=-yx/det,jx=-xy/det,jy=xx/det;
    for(let y=0;y<h;y++){const x0=ox+y*yx,y0=oy+y*yy;for(let i=0;i<w;i++)pixels[y*w+i]=base(x0+i*xx,y0+i*xy);}
    for(const g of runs)for(let root=g.first;root<=g.last;root++)for(const s of g.segs){
      const ax=(root+s.u-ox)*ix+(s.v-oy)*iy,ay=(root+s.u-ox)*jx+(s.v-oy)*jy;
      const bx=ax+s.du*ix+s.dv*iy,by=ay+s.du*jx+s.dv*jy;
      const rx=s.r*Math.hypot(ix,iy),ry=s.r*Math.hypot(jx,jy);
      let l=Math.max(0,Math.ceil(Math.min(ax,bx)-rx-1e-8)),r=Math.min(w-1,Math.floor(Math.max(ax,bx)+rx+1e-8));
      let bottom=Math.max(0,Math.ceil(Math.min(ay,by)-ry-1e-8)),top=Math.min(h-1,Math.floor(Math.max(ay,by)+ry+1e-8));
      for(let y=bottom;y<=top;y++){const x0=ox+y*yx-root-s.u,y0=oy+y*yy-s.v;
        for(let x=l;x<=r;x++){let i=y*w+x,z=hit(s,x0+xx*x,y0+xy*x);
          if(z>depth[i]){depth[i]=z;pixels[i]=s.color;}}
      }
    }
    return pixels;
  }
  return {native,prepare,makeAtlas,render,sample,reset,brush,part,edited,locate,location,extent,count:width*width,bg};
})();
