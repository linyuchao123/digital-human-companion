const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const html=fs.readFileSync('integrated.html','utf8');
const timingStart=html.indexOf('let _inputTimings=');
const timingCode=html.slice(timingStart,html.indexOf('function _percentile(',timingStart));
const chatStart=html.indexOf('function chatSend(');
const chatCode=html.slice(chatStart,html.indexOf('function handleChatMessage(',chatStart));
const textStart=html.indexOf('function sendText(');
const textCode=html.slice(textStart,html.indexOf('/* ═══════════ 窗口自适应',textStart));
function setup(socket){
  const sent=[],bubbles=[];
  const input={value:' 我的草稿 ',focused:false,focus(){this.focused=true;}};
  const feedback={hidden:true,textContent:''};
  const document={getElementById(id){return id==='text-input'?input:feedback;}};
  const context=vm.createContext({document,chatWs:socket,WebSocket:{OPEN:1},Map,performance:{now:()=>100},
    globalThis:{crypto:{randomUUID:()=> '123e4567-e89b-42d3-a456-426614174000'}},
    addBubble(role,text){bubbles.push({role,text});}});
  vm.runInContext(timingCode+chatCode+textCode,context);
  return {context,input,feedback,bubbles,sent};
}
for(const socket of [null,{readyState:0},{readyState:2},{readyState:3},{readyState:1,send(){throw Error('private transport detail');}}]){
  const state=setup(socket);
  assert.equal(vm.runInContext('sendText()',state.context),false);
  assert.equal(state.input.value,' 我的草稿 ');
  assert.equal(state.bubbles.length,0);
  assert.equal(state.feedback.hidden,false);assert(state.input.focused);
  assert(!state.feedback.textContent.includes('private transport detail'));
}
{
  const payloads=[];
  const state=setup({readyState:1,send(payload){payloads.push(JSON.parse(payload));}});
  assert.equal(vm.runInContext('sendText()',state.context),true);
  assert.deepEqual(payloads,[{type:'text_input',text:'我的草稿',request_id:'123e4567-e89b-42d3-a456-426614174000'}]);
  assert.deepEqual(state.bubbles,[{role:'user',text:'我的草稿'}]);
  assert.equal(state.input.value,'');assert.equal(state.feedback.hidden,true);
  assert.equal(vm.runInContext('sendText()',state.context),false);
  assert.equal(payloads.length,1,'Empty draft never submits');
}
{
  const state=setup(null);
  vm.runInContext('sendText()',state.context);
  const payloads=[];
  state.context.chatWs={readyState:1,send(payload){payloads.push(JSON.parse(payload));}};
  assert.equal(vm.runInContext('sendText()',state.context),true);
  assert.equal(payloads.length,1,'Only explicit retry, no background resend');
  assert.equal(state.feedback.hidden,true);assert.equal(state.feedback.textContent,'');
}
console.log('Text-send disconnected, failure, draft preservation and explicit-retry checks passed');
