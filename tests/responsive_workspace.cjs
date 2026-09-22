const assert=require('node:assert/strict'),fs=require('node:fs');
const html=fs.readFileSync('integrated.html','utf8');
assert(html.includes('--panel-width:clamp(320px,22vw,400px)'));
assert(html.includes("setProperty('--panel-width',newW+'px')"),'拖动宽度必须联动左右面板');
assert(html.includes('#sidebar.open ~ #sidebar-toggle{left:calc(var(--panel-gap) + var(--panel-width) - 1px)}'),
       '侧边栏收束按钮必须贴合面板右边缘');
assert(html.includes('#sidebar-toggle{position:fixed;left:0;top:50%;transform:translateY(-50%)'),
       '侧边栏收束按钮必须保持垂直居中');
assert(html.includes('width:20px;height:48px')&&html.includes('body.app-connected #sidebar-toggle'),
       '侧栏把手必须轻量化并仅在连接成功后显示');
assert(html.includes('.cp-controls{display:grid;grid-template-columns:'),'会话标题和操作区必须使用稳定的两行布局');
assert(html.includes("setEmotionStyleVisibility(mode==='emotional')"),'情感风格切换必须保留头部布局空间');
assert(!html.includes('renameCurrentSession()'),'不再展示会话改名入口');
assert.equal((html.match(/data-tts-voice-select/g)||[]).length,3,'顶栏与朗读面板必须共享音色同步标记和同步查询');
assert(html.includes('id="topbar-tts-voice-select"'),'设置与工具必须直接提供音色选择');
assert(html.includes("select.onchange=()=>_syncTtsVoiceSelection(select.value)"),'两个音色选择器必须双向同步');
assert(html.includes('#topbar{position:fixed;top:0;left:0;right:0;z-index:230'),
       '顶栏层级必须高于右侧对话面板');
assert(html.includes('id="read-tool-button"')&&html.includes("switchTab('chat')"),'朗读工具必须可进入并返回对话');
assert(!html.includes("getElementById('tab-chat')")&&!html.includes("getElementById('tab-read')"),
       '朗读切换不可再访问已移除的旧页签');
assert(html.includes('_live2dBaseSize={width:live2dModel.width,height:live2dModel.height}'));
assert(html.includes('wrap.clientWidth/_live2dBaseSize.width'),'缩放必须使用原始模型尺寸');
assert.equal((html.match(/Math\.min\(scW,scH\)\*LIVE2D_VIEW_SCALE/g)||[]).length,2,
       '首次加载和窗口变化必须共用标准桌面比例');
assert(html.includes("if(live2dModel){resizeLive2D();return;}"),'再次连接也必须恢复默认数字人比例');
assert(html.includes('new ResizeObserver(()=>resizeLive2D())'));
assert(html.includes('viewport-fit=cover'),'iOS 刘海屏必须启用安全区视口');
assert(html.includes('--mobile-chat:clamp(250px,40dvh,380px)'),'手机端必须为数字人与聊天分配独立纵向区域');
assert(html.includes('#mobile-panel-backdrop.show{display:block}'),'手机历史会话必须使用带遮罩的独立抽屉');
assert(html.includes("if(_authToken&&!isMobileLayout()&&!_sidebarOpen) toggleSidebar()"),'连接后不得在手机上自动遮挡主界面');
assert(html.includes("if(!isMobileLayout()&&!_sidebarOpen) toggleSidebar()"),'手机登录后不得自动展开历史栏');
assert(html.includes('@supports not ((backdrop-filter:blur(1px))'),'不支持毛玻璃的安卓浏览器必须使用不透明背景回退');
assert(html.includes('@media(max-width:900px) and (max-height:500px) and (orientation:landscape)'),
       '手机横屏必须按短屏设备识别，不能只依赖竖屏宽度');
assert(html.includes("(max-width:700px), (max-width:900px) and (max-height:500px)"),
       '脚本必须与 CSS 使用一致的手机断点');
assert(html.includes('@media(max-width:760px),(max-width:900px) and (max-height:500px)'),
       '手机横屏登录页也必须切换为单栏布局');
assert(!html.includes('id="auth-guest"'),'登录页不得再提供游客体验入口');
assert(html.includes('id="mobile-sidebar-close"'),'移动历史抽屉需要独立关闭按钮，不能覆盖新建按钮');
assert(html.includes("if(isMobileLayout()&&_sidebarOpen)toggleSidebar();"),'选择历史与新建对话后必须收起手机抽屉');
assert(html.includes('.topbar-secondary{display:block!important'),'手机端必须保留设置与工具入口');
const settingsMenu=html.match(/<details class="topbar-secondary">[\s\S]*?<\/details>/)?.[0]||'';
assert(!settingsMenu.includes('使用指南与反馈'),'使用指南不能继续挤在设置菜单中');
assert(!settingsMenu.includes('运营后台'),'运营后台不能继续挤在设置菜单中');
assert(html.includes('id="guide-btn"')&&html.includes('class="tb-chip topbar-primary"'),'顶栏必须提供醒目的独立使用指南入口');
assert(html.includes('id="developer-btn" style="display:none"'),'运营后台必须是默认隐藏的独立顶栏入口');
assert(html.includes("if(!_profile?.guide_seen)setTimeout(openFirstUseGuide,120)"),'未完成引导的新账号登录后必须自动打开使用指南');
assert(html.includes("fetch('/api/profile/guide/seen'"),'首次引导关闭状态必须保存到账号而不是当前浏览器');
assert(!html.includes('first-connection-guide'),'使用指南不能继续依赖首次连接或本机存储判断');
assert(html.includes('#guide-dialog{position:fixed;inset:0;margin:auto'),'使用指南必须稳定居中显示');
assert(html.includes('#camera-video{width:130px;aspect-ratio:1;object-fit:cover'),'手机摄像头预览必须固定为右上角小方窗');
assert(html.includes('id="guide-dialog"')&&html.includes('📘 使用指南与反馈'),'桌面与手机必须共享使用指南入口');
assert(html.includes('https://github.com/linyuchao123/digital-human-companion'),'使用指南必须包含项目仓库');
assert(html.includes('https://x.com/xiaolinyx123'),'使用指南必须包含作者 X 链接');
assert(html.includes("fetch('/api/feedback'"),'反馈信箱必须提交到服务端');
assert(html.includes('id="hold-to-talk"')&&html.includes('按住说话，松开发送'),'手机输入栏必须提供微信式按住说话入口');
assert(html.includes("holdToTalkButton.addEventListener('pointerdown',startHoldToTalk)"),'按住动作必须开始录音');
assert(html.includes("holdToTalkButton.addEventListener('pointerup',event=>finishHoldToTalk(event,true))"),'松手必须结束并发送录音');
assert(html.includes('if(sessionId!==_currentDbSession||epoch!==_sessionSwitchEpoch)return;')||
       html.includes("if(sessionId!==_currentDbSession)return;\n    if(epoch!==_sessionSwitchEpoch)return;"));
assert(html.includes('async function loadOlderMessages()'));
assert(html.includes("area.scrollTop=area.scrollHeight-previousHeight"),'加载早期消息后保持阅读位置');
console.log('稳定会话头部、侧栏收束、朗读音色、响应式缩放与历史增量加载测试通过');
