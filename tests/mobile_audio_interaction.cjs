const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

const html=fs.readFileSync('integrated.html','utf8');
const start=html.indexOf('function _ensureAudioCtx()');
const end=html.indexOf('function _stopAudio()',start);
const helperCode=`let _audioCtx=null;\n${html.slice(start,end)}`;

let resumeCalls=0,sourceStarts=0,remotePlayCalls=0;
class AudioContextMock{
  constructor(){this.state='suspended';this.sampleRate=48000;this.destination={};}
  async resume(){resumeCalls++;this.state='running';}
  createBuffer(){return {};}
  createBufferSource(){return {connect(){},disconnect(){},start(){sourceStarts++;}};}
}
const remoteAudio={srcObject:{},play:async()=>{remotePlayCalls++;}};
const context=vm.createContext({
  window:{AudioContext:AudioContextMock},
  document:{getElementById:id=>id==='wav2lip-audio'?remoteAudio:null},
  console,Promise,Math
});
vm.runInContext(helperCode,context);

(async()=>{
  const unlocked=await vm.runInContext('_unlockAudioPlayback()',context);
  assert.equal(unlocked.state,'running','A real user gesture resumes the mobile AudioContext');
  assert.equal(resumeCalls,1);
  assert.equal(sourceStarts,1,'A silent frame is started inside the user gesture');
  assert.equal(remotePlayCalls,1,'Existing Wav2Lip remote audio is retried inside the gesture');
  const resumed=await vm.runInContext('_resumeAudioPlayback()',context);
  assert.equal(resumed.state,'running');
  assert.equal(resumeCalls,1,'An already running context is not needlessly resumed');
  assert(html.includes("if(ttsSpeaking||_replySpeech.running||_replySpeech.queued.length||_streamReply.trace)beginStreamReply(null)"),
    'Holding to talk interrupts queued digital-human speech');
  assert(html.includes("window.addEventListener('pointerup'"),'Global release fallback prevents stuck mobile recording');
  console.log('移动端音频解锁、远端音频重试、播报打断与全局松手保护测试通过');
})().catch(error=>{console.error(error);process.exitCode=1;});
