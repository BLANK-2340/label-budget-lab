'use strict';
const BudgetEngine=(()=>{
  function rows(values,labels=true){
    if(!Array.isArray(values)||!values.length||values.length>512||new Set(values.map(p=>p.id)).size!==values.length)throw Error('Unique nonempty bounded rows required');
    for(const p of values)if(typeof p.id!=='string'||!p.id||!Array.isArray(p.x)||p.x.length!==2||!p.x.every(v=>Number.isFinite(v)&&Math.abs(v)<=1)||(labels&&p.y!==0&&p.y!==1))throw Error('Invalid point or label');
  }
  function weights(w){if(!Array.isArray(w)||w.length!==3||!w.every(Number.isFinite))throw Error('Finite three-parameter model required');}
  function sigmoid(z){if(!Number.isFinite(z))throw Error('Finite logit required');return z>=0?1/(1+Math.exp(-z)):Math.exp(z)/(1+Math.exp(z));}
  function logit(w,x){return w[0]*x[0]+w[1]*x[1]+w[2];}
  function objective(w,data,l2=.02){
    weights(w);rows(data);if(!Number.isFinite(l2)||l2<0)throw Error('Nonnegative finite L2 required');
    let loss=0;const grad=[0,0,0];
    for(const p of data){const z=logit(w,p.x),residual=sigmoid(z)-p.y;loss+=Math.max(z,0)-z*p.y+Math.log1p(Math.exp(-Math.abs(z)));grad[0]+=residual*p.x[0];grad[1]+=residual*p.x[1];grad[2]+=residual;}
    return {loss:loss/data.length+l2*(w[0]**2+w[1]**2)/2,grad:grad.map((g,j)=>g/data.length+(j<2?l2*w[j]:0))};
  }
  function fit(data){rows(data);let w=[0,0,0];for(let t=0;t<400;t++){const {grad}=objective(w,data);w=w.map((v,j)=>v-.5*grad[j]);}weights(w);return w;}
  function score(w,data){weights(w);rows(data);const m={tp:0,tn:0,fp:0,fn:0};for(const p of data){const predicted=logit(w,p.x)>=0;const key=predicted?(p.y?'tp':'fp'):(p.y?'fn':'tn');m[key]++;}return {...m,accuracy:(m.tp+m.tn)/data.length,log_loss:objective(w,data,0).loss};}
  function selectLabeled(splits,ids){
    rows(splits.initial);rows(splits.pool);const all=[...splits.initial,...splits.pool];
    if(new Set(all.map(p=>p.id)).size!==all.length||!Array.isArray(ids)||new Set(ids).size!==ids.length||![8,16,24,32,40].includes(ids.length))throw Error('Invalid budget or split overlap');
    const map=new Map(all.map(p=>[p.id,p]));if(ids.some(id=>!map.has(id))||splits.initial.some(p=>!ids.includes(p.id)))throw Error('Query outside annotation pool or missing initial labels');
    return ids.map(id=>{const p=map.get(id);return {id:p.id,x:[...p.x],y:p.y};});
  }
  function predict(w,x){weights(w);rows([{id:'probe',x}],false);return sigmoid(logit(w,x));}
  return {rows,objective,fit,score,selectLabeled,predict};
})();
if(typeof module!=='undefined')module.exports=BudgetEngine;
