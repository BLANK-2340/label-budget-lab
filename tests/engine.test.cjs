const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const E=require('../docs/engine.js'),data=JSON.parse(fs.readFileSync('docs/evidence.json','utf8'));
test('120 browser refits match Python parameters, confusion, accuracy and loss',()=>{
  let n=0;for(const run of data.report.runs){const d=data.datasets.find(d=>d.seed===run.seed&&d.regime===run.regime);
    for(const c of run.curve){const labeled=E.selectLabeled(d.splits,c.labeled_ids),w=E.fit(labeled),m=E.score(w,d.splits.test);
      w.forEach((v,i)=>assert.ok(Math.abs(v-c.weights[i])<1e-12));for(const k of ['tp','tn','fp','fn','accuracy'])assert.equal(m[k],c.test[k]);assert.ok(Math.abs(m.log_loss-c.test.log_loss)<1e-12);n++;}}
  assert.equal(n,120);
});
test('stable known loss, finite differences and invalid inputs',()=>{
  const rows=[{id:'a',x:[1,0],y:0}];assert.equal(E.objective([1000,0,0],rows,0).loss,1000);
  const w=[.4,-.9,.1],labeled=[{id:'a',x:[.2,-.8],y:1},{id:'b',x:[-.7,.3],y:0}],g=E.objective(w,labeled).grad;
  for(let j=0;j<3;j++){const plus=[...w],minus=[...w];plus[j]+=1e-5;minus[j]-=1e-5;assert.ok(Math.abs((E.objective(plus,labeled).loss-E.objective(minus,labeled).loss)/2e-5-g[j])<1e-8);}
  for(const rows of [[],[{id:'a',x:[NaN,0],y:0}],[{id:'a',x:[0,0],y:2}]])assert.throws(()=>E.fit(rows));
  assert.throws(()=>E.predict([0,0,Infinity],[0,0]));
});
test('test IDs rejected for fit selection; flipping test labels cannot change refit',()=>{
  const d=data.datasets[0],r=data.report.runs[0],ids=r.curve[2].labeled_ids,labeled=E.selectLabeled(d.splits,ids),before=JSON.stringify(d);
  assert.throws(()=>E.selectLabeled(d.splits,[...ids.slice(0,-1),'test:000']));assert.throws(()=>E.selectLabeled(d.splits,[...ids.slice(0,-1),ids[0]]));
  const w=E.fit(labeled),flipped=d.splits.test.map(p=>({...p,y:1-p.y}));assert.equal(E.score(w,d.splits.test).accuracy+E.score(w,flipped).accuracy,1);
  assert.deepEqual(E.fit(E.selectLabeled({...d.splits,test:flipped},ids)),w);assert.equal(JSON.stringify(d),before);
});
test('UI refits controls, overlays test points, computes probe and orients probability field',async()=>{
  const elements={},ctx=new Proxy({}, {get:()=>()=>{}});let probeInputs=[];
  for(const id of ['regime','seed','policy','budget','refit','testpoints','boundary','curve','status','metrics','aggregate','probe','legend'])elements[id]={value:'',checked:false,disabled:true,textContent:'',listeners:{},getContext:()=>ctx,getBoundingClientRect:()=>({left:0,top:0,width:660,height:520}),addEventListener(k,f){this.listeners[k]=f;},replaceChildren(){}};
  Object.assign(elements.regime,{value:'linear'});elements.seed.value='42';elements.policy.value='random';elements.budget.value='40';
  const sandbox={document:{getElementById:id=>elements[id],createElement:()=>({style:{}})},BudgetEngine:{...E,predict:(w,x)=>{probeInputs.push(x);return E.predict(w,x);}},fetch:async()=>({ok:true,json:async()=>structuredClone(data)}),console};
  vm.runInNewContext(fs.readFileSync('docs/app.js','utf8'),sandbox);await new Promise(r=>setImmediate(r));
  assert.equal(elements.refit.disabled,false);assert.match(elements.status.textContent,/Python parity passed/);assert.ok(probeInputs[0][0]<-.9&&probeInputs[0][1]>.9);
  elements.budget.value='8';elements.budget.listeners.change();assert.match(elements.metrics.textContent,/8 labels/);
  elements.regime.value='curved';elements.policy.value='uncertainty_diversity';elements.policy.listeners.change();assert.match(elements.status.textContent,/parity passed/);
  const before=elements.metrics.textContent;elements.testpoints.checked=true;elements.testpoints.listeners.change();assert.equal(elements.metrics.textContent,before);
  elements.boundary.listeners.click({clientX:330,clientY:250});assert.match(elements.probe.textContent,/Live prediction at \(0.000, 0.000\)/);
  elements.refit.listeners.click();assert.match(elements.status.textContent,/parity passed/);
});
