const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const html=fs.readFileSync('integrated.html','utf8');
const start=html.indexOf('function renderKnowledgeFollowups(');
const code=html.slice(start,html.indexOf('function renderKnowledgeSources(',start));
function element(tag){return {tag,children:[],attributes:{},handlers:{},appendChild(child){child.parentElement=this;this.children.push(child);},setAttribute(k,v){this.attributes[k]=v;},addEventListener(k,v){this.handlers[k]=v;}};}
function setup(){
  const row=element('row'),bubble=element('bubble');row.appendChild(bubble);
  const input={value:'',focused:false,focus(){this.focused=true;}};
  let latest=row,thinking=false;
  const document={createElement:element,querySelector(selector){return selector.endsWith('.bubble')?bubble:latest;},getElementById(id){return id==='text-input'?input:{classList:{contains(){return thinking;}}};}};
  const context=vm.createContext({document});vm.runInContext(code,context);
  return {context,bubble,input,setLatest(value){latest=value;},setThinking(value){thinking=value;}};
}
function render(state,sources=[{source:'NHS',excerpt:'参考知识'}]){state.context.sources=sources;vm.runInContext('renderKnowledgeFollowups(sources)',state.context);}
{
  const state=setup();for(const value of [null,[],['invalid'],[{}],[{excerpt:''}]])render(state,value);
  assert.equal(state.bubble.children.length,0,'No suggestions without usable knowledge');
}
{
  const state=setup();render(state);
  const box=state.bubble.children[0],buttons=box.children.filter(e=>e.tag==='button');
  assert.equal(buttons.length,3);
  assert(buttons.every(e=>e.type==='button'));
  buttons[1].handlers.click();assert.equal(state.input.value,'能举个例子吗');assert(state.input.focused);
  assert.equal(box.children.at(-1).attributes.role,'status');
  state.input.value='自己的草稿';buttons[0].handlers.click();assert.equal(state.input.value,'自己的草稿');
  state.input.value='';state.setThinking(true);buttons[0].handlers.click();assert.equal(state.input.value,'');
  state.setThinking(false);state.setLatest(element('different-topic'));buttons[0].handlers.click();assert.equal(state.input.value,'');
}
assert(!code.includes('innerHTML'),'Suggestions must not parse HTML');
assert(!code.includes('chatSend(')&&!code.includes('sendText(')&&!code.includes('fetch('),'Selection never submits automatically');
assert(html.includes('renderKnowledgeFollowups(msg.knowledge_sources)'));
console.log('Knowledge follow-up draft, stale-topic and busy-state checks passed');
