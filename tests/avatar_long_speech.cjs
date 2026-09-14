const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const html=fs.readFileSync('integrated.html','utf8');
const start=html.indexOf('let _gesture=null,_gestureLast=0;');
const end=html.indexOf('let _wakeUp=null;',start);
const ctx=vm.createContext({_ttsGeneration:1,ttsSpeaking:true,connected:true,
  _useBrowserTTS:false,_audioCtx:{currentTime:0,state:'running'},_audioSrc:{},
  _streamSources:new Set(),document:{hidden:false},performance:{now:()=>0},
  isPlayingMotion:false,_wakeUp:null,_getLipSyncTarget:()=>0.3});
vm.runInContext(html.slice(start,end),ctx);
vm.runInContext("var calls=[];function startGesture(name){calls.push(name);_gesture={name};}startSpeechChoreography('我会陪着你，慢慢来',1)",ctx);
ctx._audioCtx.currentTime=7;vm.runInContext('tickSpeechChoreography()',ctx);
assert.equal(vm.runInContext('calls.length',ctx),0,'Short speech does not add motion');
ctx._audioCtx.currentTime=8;vm.runInContext('tickSpeechChoreography()',ctx);
assert.equal(vm.runInContext('calls[0]',ctx),'Comfort');
ctx._audioCtx.currentTime=20;vm.runInContext('tickSpeechChoreography()',ctx);
assert.equal(vm.runInContext('calls.length',ctx),1,'Never interrupt active gestures');
vm.runInContext('_gesture=null',ctx);ctx.document.hidden=true;
vm.runInContext('tickSpeechChoreography()',ctx);
assert.equal(vm.runInContext('calls.length',ctx),1,'Hidden pages do not animate');
ctx.document.hidden=false;ctx._audioCtx.state='suspended';
vm.runInContext('tickSpeechChoreography()',ctx);
assert.equal(vm.runInContext('calls.length',ctx),1,'Paused audio does not animate');
ctx._audioCtx.state='running';ctx._getLipSyncTarget=()=>0;
vm.runInContext('tickSpeechChoreography()',ctx);
assert.equal(vm.runInContext('calls.length',ctx),1,'Buffer silence does not animate');
ctx._getLipSyncTarget=()=>0.3;
for(let second=20;second<10000;second++){
  ctx._audioCtx.currentTime=second;
  vm.runInContext('_gesture=null;tickSpeechChoreography()',ctx);
}
assert.equal(vm.runInContext('calls.length',ctx),3,'Long-running speech has bounded gestures');
ctx._ttsGeneration=2;vm.runInContext('tickSpeechChoreography()',ctx);
assert.equal(vm.runInContext('_speechChoreography',ctx),null,'Cancellation clears the plan');
vm.runInContext("startSpeechChoreography('我们想想可以怎么整理',2)",ctx);
assert.equal(vm.runInContext('_speechChoreography.motion',ctx),'Think');
ctx.connected=false;vm.runInContext('tickSpeechChoreography()',ctx);
assert.equal(vm.runInContext('_speechChoreography',ctx),null,'Disconnect clears the plan');
assert(html.includes('function _stopAudio(){\n  _ttsGeneration++;\n  _speechChoreography=null;'),'Stopping playback clears choreography');
console.log('长语音节奏、动作互斥、静音暂停、长期有界与取消清理测试通过');
