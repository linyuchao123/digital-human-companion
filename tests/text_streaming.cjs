const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const html=fs.readFileSync('integrated.html','utf8');
const start=html.indexOf('let _streamReply={'),end=html.indexOf('function showThinking(',start);
let created=0,bubble=null;
const area={scrollTop:0,scrollHeight:100};
const context=vm.createContext({document:{getElementById:id=>id==='chat-area'?area:{textContent:''},
  querySelector:()=>bubble,createElement:()=>({})},addBubble:()=>{
    created++;bubble={isConnected:true,textContent:'',setAttribute(){},appendChild(){},dataset:{}};
  }});
vm.runInContext(html.slice(start,end),context);
vm.runInContext("beginStreamReply('a');appendStreamReply({trace_id:'a',text:'<script>'});appendStreamReply({trace_id:'a',text:'你好'});",context);
assert.equal(created,1,'All tokens use one bubble');
assert.equal(bubble.textContent,'<script>你好','Tokens remain plain text');
vm.runInContext("appendStreamReply({trace_id:'old',text:'迟到'});finishStreamReply({trace_id:'a',emotion_label:'平静'},'完整回答');",context);
assert.equal(bubble.textContent,'完整回答');assert.equal(created,1,'Final response does not duplicate the bubble');
vm.runInContext("appendStreamReply({trace_id:'a',text:'尾部迟到'});",context);
assert.equal(bubble.textContent,'完整回答');
vm.runInContext("beginStreamReply('b');appendStreamReply({trace_id:'b',text:'新会话'});beginStreamReply(null);appendStreamReply({trace_id:'b',text:'旧会话'});",context);
assert.equal(bubble.textContent,'新会话','Session cleanup rejects old deltas');
console.log('文本真实增量单气泡、纯文本渲染、最终替换与迟到隔离测试通过');
