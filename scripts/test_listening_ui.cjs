// DOM-level application test with simulated playback, not a human listening result.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync('.artifacts/blind_listening_v1/participant/index.html','utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const ids = [...html.matchAll(/id="([^"]+)"/g)].map(m=>m[1]);
const elements = Object.fromEntries(ids.map(id=>[id,{hidden:false,value:'',disabled:false,
  classList:{remove(){},toggle(){}},events:{},addEventListener(k,f){this.events[k]=f;},pause(){}}]));
const buttons = ['tracked','wheeled','unsure'].map(choice=>({dataset:{choice},classList:{remove(){},toggle(){}}}));
const storage = new Map(); let exported=null;
const context = {document:{getElementById:id=>elements[id],querySelectorAll:()=>buttons,
  createElement:()=>({click(){}})},localStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)},
  performance:{now:()=>100},Date,JSON,Number,Blob:class{constructor(parts){exported=JSON.parse(parts[0]);}},
  URL:{createObjectURL:()=> 'blob:unit-test',revokeObjectURL(){}},setTimeout:f=>f()};
vm.runInNewContext(script,context);
elements.familiar.value='No'; elements.start.onclick();
assert.equal(elements.test.hidden,false);
for(let i=0;i<28;i++){
  assert.equal(elements.next.disabled,true);
  buttons[2].onclick(); elements.confidence.value='1'; elements.confidence.onchange();
  assert.equal(elements.next.disabled,true);
  elements.player.events.play(); assert.equal(elements.next.disabled,false);
  elements.next.onclick();
}
assert.equal(elements.done.hidden,false);
elements.export.onclick();
assert.equal(exported.responses.length,28);
assert(exported.responses.every(r=>r.choice==='unsure' && r.plays===1));
assert.equal(storage.size,1);
vm.runInNewContext(script,{...context}); elements.start.onclick();
assert.equal(elements.done.hidden,false);
console.log('PASS: 28 simulated UI trials, gating, local save, resume, JSON export. No human results saved.');
