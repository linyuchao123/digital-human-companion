const assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs');
const elements=new Map(),events={},sent=[];
const el=id=>{if(!elements.has(id)) elements.set(id,{textContent:'',hidden:false,readyState:2,classList:{add(){},remove(){}},play:async()=>{},srcObject:null});return elements.get(id);};
let acquire,stopped=0;
const context={document:{hidden:false,getElementById:el,addEventListener:(k,f)=>events[k]=f},navigator:{mediaDevices:{getUserMedia:()=>new Promise(r=>acquire=r)}},
  chatWsReady:true,chatSend:m=>{sent.push(m);return true;},confirm:()=>true,setTimeout,clearTimeout,
  crypto:require('node:crypto'),performance,createImageBitmap:async()=>({close(){}}),Worker:class {postMessage(){} terminate(){}},};
context.window=context;context.addEventListener=(k,f)=>events[k]=f;
vm.createContext(context);vm.runInContext(fs.readFileSync('static/mediapipe/camera-controller.js','utf8'),context);
(async()=>{
  const pending=context.toggleCameraPerception();
  context.stopCameraPerception();
  acquire({getTracks:()=>[{stop:()=>stopped++}]});await pending;
  assert.equal(stopped,1);assert.equal(el('camera-video').srcObject,null);
  const ready=context.toggleCameraPerception();
  const track={stop:()=>stopped++};
  acquire({getTracks:()=>[track],getVideoTracks:()=>[track]});await ready;
  assert.equal(sent.at(-1).type,'vision_control');assert.equal(sent.at(-1).enabled,true);
  context.document.hidden=true;events.visibilitychange();
  assert.equal(sent.at(-1).enabled,false);assert.equal(el('camera-video').srcObject,null);
  assert.equal(stopped,2);
  context.navigator.mediaDevices.getUserMedia=async()=>{throw Object.assign(Error(),{name:'NotAllowedError'});};
  await context.toggleCameraPerception();assert.match(el('camera-status').textContent,/权限被拒绝/);
  const src=fs.readFileSync('static/mediapipe/camera-controller.js','utf8');
  assert(!src.includes('toDataURL'));assert(!src.includes('MediaRecorder'));assert(!src.includes('fetch('));
  console.log('摄像头异步取消、后台关闭、权限拒绝、无图像上传测试通过');
})().catch(e=>{console.error(e);process.exitCode=1;});
