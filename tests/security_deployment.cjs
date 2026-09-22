const assert=require('node:assert/strict'),fs=require('node:fs');
const nginx=fs.readFileSync('infra/nginx/digital-xinyu.conf','utf8');
const defaultHttp=nginx.match(/server \{\s*listen 80 default_server;[\s\S]*?\n\}/)?.[0]||'';
assert(defaultHttp.includes('return 301 https://xiaolinyx.cloud$request_uri;'),'IP 与未知 Host 的 HTTP 入口必须跳转到正式 HTTPS 域名');
assert(!defaultHttp.includes('proxy_pass'),'HTTP 默认入口不得代理登录和业务请求');
assert((nginx.match(/Strict-Transport-Security/g)||[]).length>=2,'HTTPS 域名和 www 跳转均应启用 HSTS');
console.log('HTTPS 统一入口、IP 跳转与 HSTS 配置测试通过');
