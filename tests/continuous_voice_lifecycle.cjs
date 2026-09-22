const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const html=fs.readFileSync('integrated.html','utf8');
assert(!html.includes('id="continuous-voice-btn"'),'Continuous mode entry removed at user request');
const begin=html.indexOf('let _speechRec=null,'),end=html.indexOf('function _encodeVoiceWav(',begin);
const stop=html.indexOf('function stopVoiceInput('),stopEnd=html.indexOf('/* ═══════════ TTS',stop);
let resolveMic,stopped=0,sent=0;
const elements=new Map(),listeners={};
function element(id){if(!elements.has(id))elements.set(id,{classList:{add(){},remove(){}},setAttribute(){},placeholder:'',value:''});return elements.get(id);}
const ctx=vm.createContext({chatWsReady:true,ttsSpeaking:false,console:{log(){},warn(){}},
  document:{hidden:false,getElementById:element,addEventListener:(name,fn)=>listeners[name]=fn},
  fetch:async()=>({ok:true,json:async()=>({asr:{available:true,provider:'qwen_cloud'}})}),
  navigator:{mediaDevices:{getUserMedia:()=>new Promise(resolve=>resolveMic=resolve)}},
  setTimeout:()=>1,clearTimeout(){},alert(){},confirm:()=>true,_stopAudio(){},
  _resumeAudioPlayback:async()=>({sampleRate:48000,createMediaStreamSource:()=>({connect(){},disconnect(){}}),
    createScriptProcessor:()=>({connect(){},disconnect(){}}),createGain:()=>({gain:{},connect(){},disconnect(){}})}),
  chatSend:()=>sent++,Float32Array,Math});
vm.runInContext(html.slice(begin,end)+html.slice(stop,stopEnd),ctx);
(async()=>{
  vm.runInContext('_continuousVoice=true;var pending=startVoiceInput()',ctx);
  await new Promise(resolve=>setImmediate(resolve));
  assert(resolveMic,'Permission request reached');
  vm.runInContext('disableContinuousVoice()',ctx);
  resolveMic({getTracks:()=>[{stop:()=>stopped++}]});
  await vm.runInContext('pending',ctx);
  assert.equal(stopped,1,'Late permission acquisition releases its stream');
  assert.equal(vm.runInContext('_isListening',ctx),false);
  assert.equal(sent,0,'Cancelled recording never uploads');
  vm.runInContext('_continuousVoice=true;pending=startVoiceInput()',ctx);
  await new Promise(resolve=>setImmediate(resolve));
  resolveMic({getTracks:()=>[{stop:()=>stopped++}]});await vm.runInContext('pending',ctx);
  assert.equal(vm.runInContext('_isListening',ctx),true);
  ctx.document.hidden=true;listeners.visibilitychange();
  assert.equal(stopped,2,'Backgrounding releases the microphone');
  assert.equal(vm.runInContext('_continuousVoice',ctx),false);
  assert.equal(vm.runInContext('_voiceChunks.length',ctx),0);
  ctx.document.hidden=false;
  vm.runInContext('pending=startVoiceInput()',ctx);
  await new Promise(resolve=>setImmediate(resolve));
  resolveMic({getTracks:()=>[{stop:()=>stopped++}]});await vm.runInContext('pending',ctx);
  vm.runInContext('toggleContinuousVoice()',ctx);
  assert.equal(stopped,3,'Switching from manual capture releases its old stream');
  await new Promise(resolve=>setImmediate(resolve));
  vm.runInContext('disableContinuousVoice()',ctx);
  resolveMic({getTracks:()=>[{stop:()=>stopped++}]});
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(stopped,4,'Late stream from a mode switch is released');
  console.log('连续录音权限迟到取消、后台释放、无旧录音发送测试通过');
})().catch(error=>{console.error(error);process.exitCode=1;});
