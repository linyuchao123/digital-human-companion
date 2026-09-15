const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const html=fs.readFileSync('integrated.html','utf8');
const timingCode=html.slice(html.indexOf('let _inputTimings='),html.indexOf('function _percentile('));
const summaryCode=html.slice(html.indexOf('function _percentile('),html.indexOf('function renderTtsMetrics('));
const adaptiveCode=html.slice(html.indexOf('function ttsInitialBufferSeconds('),html.indexOf('async function _playStreamTTS('));
let tick=100;
const ctx=vm.createContext({window:{__ttsStreamMetrics:[]},Number,Math,Map,performance:{now:()=>++tick},
  globalThis:{crypto:{randomUUID:()=> '123e4567-e89b-42d3-a456-426614174000'}}});
vm.runInContext(timingCode+summaryCode+adaptiveCode,ctx);
const requestId=ctx._newInputTiming('voice');
const timing=ctx._bindInputTrace(requestId,'trace-1');
assert.equal(timing.kind,'voice');assert.equal(vm.runInContext(`_inputTimings.has('${requestId}')`,ctx),false);
assert.equal(ctx._takeReplyTiming('trace-1'),timing);assert.equal(vm.runInContext('_inputTimings.size',ctx),0);
const rows=[
  {status:'completed',firstByteMs:80,firstAudioScheduledMs:150,inputToFirstAudioScheduledMs:1000,underruns:0},
  {status:'interrupted',firstByteMs:120,firstAudioScheduledMs:250,inputToFirstAudioScheduledMs:2000,underruns:1},
  {status:'completed',firstByteMs:100,firstAudioScheduledMs:200,inputToFirstAudioScheduledMs:1500,underruns:2},
];
const summary=ctx.ttsMetricsSummary(rows);
assert.equal(summary.samples,3);assert.equal(summary.completed,2);assert.equal(summary.medianFirstByteMs,100);
assert.equal(summary.medianInputToAudioMs,1500);assert.equal(summary.p95InputToAudioMs,2000);assert.equal(summary.totalUnderruns,3);
assert.equal(ctx.ttsInitialBufferSeconds(rows),.56,'Repeated underruns increase only the next initial buffer');
assert.equal(ctx.ttsInitialBufferSeconds([{underruns:0},{underruns:1}]),.32,'Normal network keeps original buffer');
assert(!summaryCode.includes('row.text')&&!summaryCode.includes('row.audio'),'Summary must not read content or audio');
console.log('TTS 请求关联、本机延迟汇总、P95 与弱网自适应缓冲测试通过');
