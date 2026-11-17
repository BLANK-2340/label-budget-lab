'use strict';
const $=id=>document.getElementById(id);
const colors={random:'#b6b1ff',uncertainty:'#89e6ce',diversity:'#ffbd7d',uncertainty_diversity:'#ff8ca9'};
let bundle,current;
const xy=x=>[48+(x[0]+1)*282,460-(x[1]+1)*210];
function drawBoundary(){
  const ctx=$('boundary').getContext('2d');ctx.clearRect(0,0,660,520);
  for(let ix=0;ix<60;ix++)for(let iy=0;iy<44;iy++){const p=BudgetEngine.predict(current.weights,[-1+(ix+.5)*2/60,1-(iy+.5)*2/44]);ctx.fillStyle=p>=.5?`rgba(255,168,91,${.08+.35*Math.abs(p-.5)*2})`:`rgba(90,220,202,${.08+.35*Math.abs(p-.5)*2})`;ctx.fillRect(48+ix*564/60,40+iy*420/44,564/60+1,420/44+1);}
  ctx.strokeStyle='#92aab9';ctx.lineWidth=1;ctx.strokeRect(48,40,564,420);ctx.fillStyle='#c5d4df';ctx.font='14px system-ui';ctx.fillText('x₁ : −1 → +1',260,500);ctx.fillText('x₂ : +1',48,25);ctx.fillText('−1',18,460);
  const ids=new Set(current.ids),initial=new Set(current.split.initial.map(p=>p.id));
  for(const p of current.split.pool)if(!ids.has(p.id)){const [x,y]=xy(p.x);ctx.fillStyle='#708493';ctx.beginPath();ctx.arc(x,y,2.5,0,Math.PI*2);ctx.fill();}
  if($('testpoints').checked)for(const p of current.split.test){const [x,y]=xy(p.x);ctx.fillStyle=p.y?'#ffb56e':'#75d8c3';ctx.fillRect(x-1.5,y-1.5,3,3);}
  for(const p of current.labeled){const [x,y]=xy(p.x);ctx.fillStyle=p.y?'#ffb56e':'#75d8c3';ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.fill();if(initial.has(p.id)){ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(x,y,8,0,Math.PI*2);ctx.stroke();}}
}
function drawCurve(){
  const ctx=$('curve').getContext('2d');ctx.clearRect(0,0,660,330);ctx.font='13px system-ui';ctx.lineWidth=1;ctx.strokeStyle='#41586b';ctx.fillStyle='#bfd0de';
  for(const accuracy of [0,.25,.5,.75,1]){const y=280-accuracy*240;ctx.beginPath();ctx.moveTo(48,y);ctx.lineTo(612,y);ctx.stroke();ctx.fillText((accuracy*100)+'%',8,y+4);}
  for(const budget of [8,16,24,32,40])ctx.fillText(String(budget),48+(budget-8)/32*564-8,307);
  for(const policy of Object.keys(colors)){const run=bundle.report.runs.find(r=>r.seed===current.seed&&r.regime===current.regime&&r.policy===policy);ctx.strokeStyle=colors[policy];ctx.lineWidth=policy===current.policy?3.5:1.5;ctx.beginPath();run.curve.forEach((c,i)=>{const x=48+(c.budget-8)/32*564,y=280-c.test.accuracy*240;if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();}
  ctx.strokeStyle='#fff';ctx.setLineDash([4,5]);const x=48+(current.ids.length-8)/32*564;ctx.beginPath();ctx.moveTo(x,40);ctx.lineTo(x,280);ctx.stroke();ctx.setLineDash([]);
  $('legend').replaceChildren(...Object.entries(colors).map(([name,color])=>{const span=document.createElement('span');span.style.color=color;span.textContent=name.replaceAll('_',' ')+' · ';return span;}));
}
function refit(){
  try{
    const seed=+$('seed').value,regime=$('regime').value,policy=$('policy').value,budget=+$('budget').value;
    const data=bundle.datasets.find(d=>d.seed===seed&&d.regime===regime),run=bundle.report.runs.find(r=>r.seed===seed&&r.regime===regime&&r.policy===policy),saved=run.curve.find(c=>c.budget===budget);
    const labeled=BudgetEngine.selectLabeled(data.splits,saved.labeled_ids),weights=BudgetEngine.fit(labeled),metrics=BudgetEngine.score(weights,data.splits.test);
    current={seed,regime,policy,split:data.splits,ids:saved.labeled_ids,labeled,weights,metrics};
    if(weights.some((w,i)=>Math.abs(w-saved.weights[i])>1e-10)||Math.abs(metrics.log_loss-saved.test.log_loss)>1e-10||metrics.accuracy!==saved.test.accuracy)throw Error('Browser/Python parity check failed');
    $('metrics').textContent=`Live held-out accuracy ${(metrics.accuracy*100).toFixed(2)}% · log loss ${metrics.log_loss.toFixed(5)} · ${budget} labels`;
    $('status').textContent=`Refit complete: ${budget} labels, ${104-budget} unqueried annotation points. Python parity passed.`;
    const aggregate=bundle.report.aggregate.find(r=>r.regime===regime&&r.policy===policy).curve.find(c=>c.budget===budget);
    $('aggregate').textContent=`Saved three-seed mean for this strategy/budget: ${(aggregate.accuracy_mean*100).toFixed(2)}% (sample SD ${(aggregate.accuracy_sample_stddev*100).toFixed(2)} percentage points). Not a confidence interval.`;
    $('probe').textContent='Click the plot to evaluate a new point. Probability is model output, not calibrated confidence.';drawBoundary();drawCurve();
  }catch(e){$('status').textContent='Cannot refit: '+e.message;}
}
fetch('evidence.json').then(r=>{if(!r.ok)throw Error('evidence request failed');return r.json();}).then(data=>{
  if(data.schema!==1||data.datasets.length!==6||data.report.runs.length!==24)throw Error('unexpected evidence format');bundle=data;$('refit').disabled=false;
  for(const id of ['regime','seed','policy','budget'])$(id).addEventListener('change',refit);
  $('refit').addEventListener('click',refit);$('testpoints').addEventListener('change',()=>{if(current)drawBoundary();});
  $('boundary').addEventListener('click',event=>{if(!current)return;const rect=$('boundary').getBoundingClientRect(),px=(event.clientX-rect.left)*660/rect.width,py=(event.clientY-rect.top)*520/rect.height;if(px<48||px>612||py<40||py>460)return;const x=[(px-48)/282-1,(460-py)/210-1],p=BudgetEngine.predict(current.weights,x);$('probe').textContent=`Live prediction at (${x.map(v=>v.toFixed(3)).join(', ')}): P(class 1) = ${p.toFixed(5)}. No new labels spent.`;});refit();
}).catch(e=>{$('status').textContent='Evidence unavailable: '+e.message;});
