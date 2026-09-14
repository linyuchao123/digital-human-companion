const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const html=fs.readFileSync('integrated.html','utf8');
const code=html.slice(html.indexOf('async function _playStreamTTS('),html.indexOf('const _SWEET_VOICES'));
async function fixture(chunks){
  const sources=[],timers=[];let reads=0,released=false;
  const ctx={currentTime:0,baseLatency:0.01,createAnalyser:()=>({connect(){},disconnect(){}}),
    createBuffer:(_,length,rate)=>({duration:length/rate,getChannelData:()=>new Float32Array(length)}),
    createBufferSource:()=>{const source={connect(){},disconnect(){},start(at){this.at=at;}};sources.push(source);return source;}};
  const reader={async read(){return reads<chunks.length?{value:chunks[reads++],done:false}:{done:true};},async cancel(){},releaseLock(){released=true;}};
  const context=vm.createContext({window:{},performance:{now:()=>100},console,DataView,Uint8Array,Set,
    setTimeout:(fn,delay)=>{timers.push({fn,delay});return timers.length;},clearTimeout(){},
    _ttsGeneration:0,_authToken:'',_analyser:null,_streamSources:new Set(),_speechChoreography:{},
    ttsSpeaking:false,_mouthSmooth:1,_useBrowserTTS:false,
    document:{getElementById:()=>({classList:{add(){},remove(){}}})},
    _ensureAudioCtx:()=>ctx,notifySpeechStarted(){},_stopAudio(){context._ttsGeneration++;context.ttsSpeaking=false;},
    streamChunkBytes:(rate,started)=>Math.ceil(rate*(started?0.16:0.32))*2,
    streamStartTime:(current,until,started)=>until>current+0.015?until:current+(started?0.20:0.08),
    fetch:async()=>({ok:true,headers:{get:()=>24000},body:{getReader:()=>reader}})});
  vm.runInContext(code,context);
  await context._playStreamTTS('test','same-voice',0,{signal:{},abort(){}});
  assert(released,'Reader lock released after network EOF');
  return {context,sources,timers};
}
(async()=>{
  const {context,sources,timers}=await fixture([new Uint8Array(16000),new Uint8Array(8000)]);
  assert.equal(context.ttsSpeaking,true,'Network EOF is not playback completion');
  assert(!timers.some(t=>t.delay===580),'No wall-clock playback-end timer');
  sources[0].onended();assert.equal(context.ttsSpeaking,true,'Wait for last source');
  sources[1].onended();assert.equal(context.ttsSpeaking,false);
  assert.equal(context.window.__ttsStreamMetrics[0].status,'completed');
  assert.equal(context._streamSources.size,0);
  const stale=await fixture([new Uint8Array(16000)]);
  stale.context._ttsGeneration++;stale.context.ttsSpeaking=true;
  stale.sources[0].onended();assert.equal(stale.context.ttsSpeaking,true,'Old source cannot stop new speech');
  const malformed=await fixture([new Uint8Array(16001)]);
  assert.equal(malformed.context.window.__ttsStreamMetrics[0].status,'interrupted');
  assert.equal(malformed.context.ttsSpeaking,false);
  console.log('TTS EOF, audio-clock completion, cancellation and malformed PCM checks passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
