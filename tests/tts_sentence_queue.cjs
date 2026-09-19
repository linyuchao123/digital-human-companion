const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const html=fs.readFileSync('integrated.html','utf8');
const start=html.indexOf("let _replySpeech={"),end=html.indexOf('function queueReplySpeech(',start);
const context=vm.createContext({});vm.runInContext(html.slice(start,end),context);

vm.runInContext("_replySpeech.buffer='第一句还没';",context);
assert.deepEqual(Array.from(context.splitSpeechBuffer(false)),[],'跨增量的不完整句不能提前入队');
vm.runInContext("_replySpeech.buffer+='说完。第二句继续';",context);
assert.deepEqual(Array.from(context.splitSpeechBuffer(false)),['第一句还没说完。']);
assert.equal(vm.runInContext('_replySpeech.buffer',context),'第二句继续');
assert.deepEqual(Array.from(context.splitSpeechBuffer(true)),['第二句继续'],'最终只补未入队尾部');

const long='这是一段很长的内容，'.repeat(30)+'结束。';
vm.runInContext(`_replySpeech.buffer=${JSON.stringify(long)}`,context);
const pieces=Array.from(context.splitSpeechBuffer(false));
assert(pieces.length>1);assert(pieces.every(piece=>piece.length<=180),'每个语音段不超过180字');
assert(!context.cleanSpeechText('参考[12] [MOTION:Tap]正文').includes('[12]'));
assert(!context.cleanSpeechText('参考[12] [MOTION:Tap]正文').includes('MOTION'));
console.log('分句队列、跨增量边界、长度限制与内部标记清理测试通过');
