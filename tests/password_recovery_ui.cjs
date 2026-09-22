const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const html=fs.readFileSync('integrated.html','utf8');
const code=html.slice(html.indexOf('function passwordCategoryCount('),html.indexOf('async function startSocialLogin('));
const elements=new Map();
function element(id){if(!elements.has(id))elements.set(id,{value:'',textContent:''});return elements.get(id);}
let requests=0,mode=null;
const ctx=vm.createContext({document:{getElementById:element},AbortController,Blob,setTimeout:()=>1,clearTimeout(){},
  switchAuthTab:tab=>mode=tab,fetch:async(url,options)=>{
    requests++;assert(url.endsWith('/confirm'));
    assert(!options.headers['X-Auth-Token'],'Anonymous recovery uses no login credential');
    return {ok:true,json:async()=>({message:'密码已重置'})};
  }});
vm.runInContext(code,ctx);
(async()=>{
  const button={disabled:false};
  element('recovery-email').value='person@example.org';element('recovery-code').value='123456';
  element('recovery-password').value='new-password';element('recovery-confirm').value='different';
  await ctx.confirmPasswordReset(button);assert.equal(requests,0,'Mismatch prevents request');
  element('recovery-confirm').value='new-password';
  await ctx.confirmPasswordReset(button);assert.equal(requests,1);assert.equal(mode,'login');
  for(const id of ['recovery-code','recovery-password','recovery-confirm','auth-password'])assert.equal(element(id).value,'');
  assert.equal(button.disabled,false);
  assert(!code.includes('localStorage'),'Never store reset codes or passwords');
  assert(!code.includes('innerHTML'),'Status remains plain text');
  console.log('邮箱找回密码确认、敏感字段清理与匿名请求测试通过');
})().catch(error=>{console.error(error);process.exitCode=1;});
