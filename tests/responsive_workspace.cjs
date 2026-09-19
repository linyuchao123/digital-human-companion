const assert=require('node:assert/strict'),fs=require('node:fs');
const html=fs.readFileSync('integrated.html','utf8');
assert(html.includes('--panel-width:clamp(320px,22vw,400px)'));
assert(html.includes("setProperty('--panel-width',newW+'px')"),'拖动宽度必须联动左右面板');
assert(html.includes('_live2dBaseSize={width:live2dModel.width,height:live2dModel.height}'));
assert(html.includes('wrap.clientWidth/_live2dBaseSize.width'),'缩放必须使用原始模型尺寸');
assert(html.includes('new ResizeObserver(()=>resizeLive2D())'));
assert(html.includes('if(sessionId!==_currentDbSession||epoch!==_sessionSwitchEpoch)return;')||
       html.includes("if(sessionId!==_currentDbSession)return;\n    if(epoch!==_sessionSwitchEpoch)return;"));
assert(html.includes('async function loadOlderMessages()'));
assert(html.includes("area.scrollTop=area.scrollHeight-previousHeight"),'加载早期消息后保持阅读位置');
console.log('对称面板、原始尺寸缩放、响应式观察与历史增量加载测试通过');
