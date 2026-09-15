const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const html=fs.readFileSync('integrated.html','utf8');
const code=html.slice(html.indexOf('function clearLocalAccount('),html.indexOf('function toggleSidebar('));
const elements=new Map();
const element=id=>{if(!elements.has(id))elements.set(id,{value:'private',textContent:'private',classList:{remove(){}},close(){},removeAttribute(){},focus(){}});return elements.get(id);};
let calls=0,stops=0,consent=false,ok=false;
const ctx=vm.createContext({document:{getElementById:element},_authToken:'owner-token',_currentUser:'owner',_currentDbSession:'session',_profile:{},_profileAvatar:'private',_sidebarOpen:false,
  window:{__ttsStreamMetrics:[{status:'completed'}]},_inputTimings:new Map(),
  disconnectAll(){stops++;},delCookie(){},confirm:()=>consent,AbortController,setTimeout:()=>1,clearTimeout(){},fetch:async(url,options)=>{
    calls++;assert.equal(url,'/api/profile/account/delete');assert.equal(options.headers['X-Auth-Token'],'owner-token');
    assert.equal(JSON.parse(options.body).current_password,'password');
    return {ok,json:async()=>ok?{message:'已注销'}:{error:'密码不正确'}};
  }});
vm.runInContext(code,ctx);
(async()=>{
  const button={disabled:false};
  element('account-delete-confirm').value='wrong';await ctx.deleteMyAccount(button);assert.equal(calls,0);
  element('account-delete-confirm').value='注销我的账号';element('profile-current-password').value='password';
  await ctx.deleteMyAccount(button);assert.equal(calls,0,'Cancel means no destructive request');
  consent=true;await ctx.deleteMyAccount(button);assert.equal(stops,0);assert.equal(ctx._authToken,'owner-token');
  ok=true;await ctx.deleteMyAccount(button);assert.equal(stops,1);assert.equal(ctx._authToken,'');assert.equal(ctx._currentDbSession,'');
  for(const id of ['profile-current-password','account-delete-confirm','recovery-code','recovery-password'])assert.equal(element(id).value,'');
  for(const id of ['chat-area','memory-list','agent-debug-content','profile-email-state'])assert.equal(element(id).textContent,'');
  assert.equal(button.disabled,false);console.log('注销二次确认、失败保护与本地资源清理测试通过');
})().catch(error=>{console.error(error);process.exitCode=1;});
