const assert=require('node:assert/strict'),fs=require('node:fs');
const html=fs.readFileSync('integrated.html','utf8');
assert(html.includes('--panel-width:clamp(320px,22vw,400px)'));
assert(html.includes("setProperty('--panel-width',newW+'px')"),'拖动宽度必须联动左右面板');
assert(html.includes('#sidebar.open ~ #sidebar-toggle{left:calc(var(--panel-gap) + var(--panel-width) - 1px)}'),
       '侧边栏收束按钮必须贴合面板右边缘');
assert(html.includes('.cp-controls{display:grid;grid-template-columns:'),'会话标题和操作区必须使用稳定的两行布局');
assert(html.includes("setEmotionStyleVisibility(mode==='emotional')"),'情感风格切换必须保留头部布局空间');
assert(!html.includes('renameCurrentSession()'),'不再展示会话改名入口');
assert.equal((html.match(/id="tts-voice-select"/g)||[]).length,1,'页面只能有一个可持久化音色选择器');
assert(html.includes('id="read-tool-button"')&&html.includes("switchTab('chat')"),'朗读工具必须可进入并返回对话');
assert(!html.includes("getElementById('tab-chat')")&&!html.includes("getElementById('tab-read')"),
       '朗读切换不可再访问已移除的旧页签');
assert(html.includes('_live2dBaseSize={width:live2dModel.width,height:live2dModel.height}'));
assert(html.includes('wrap.clientWidth/_live2dBaseSize.width'),'缩放必须使用原始模型尺寸');
assert(html.includes('new ResizeObserver(()=>resizeLive2D())'));
assert(html.includes('if(sessionId!==_currentDbSession||epoch!==_sessionSwitchEpoch)return;')||
       html.includes("if(sessionId!==_currentDbSession)return;\n    if(epoch!==_sessionSwitchEpoch)return;"));
assert(html.includes('async function loadOlderMessages()'));
assert(html.includes("area.scrollTop=area.scrollHeight-previousHeight"),'加载早期消息后保持阅读位置');
console.log('稳定会话头部、侧栏收束、朗读音色、响应式缩放与历史增量加载测试通过');
