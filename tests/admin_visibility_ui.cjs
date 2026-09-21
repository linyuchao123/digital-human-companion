const assert=require('node:assert/strict'),fs=require('node:fs');
const main=fs.readFileSync('integrated.html','utf8');
const admin=fs.readFileSync('admin.html','utf8');

assert.match(main,/id="developer-btn" style="display:none"/,'运营后台入口默认必须隐藏');
assert.match(main,/_profile\?\.role==='admin'\?'inline-flex':'none'/,'仅管理员登录后显示入口');
assert.match(main,/window\.location\.href='\/admin'/,'入口必须进入服务端鉴权页面');
assert.match(admin,/仅管理员可见/);
assert.match(admin,/对话轮次/);
assert.match(admin,/估算费用/);
assert.doesNotMatch(admin,/<th>聊天内容<\/th>/,'运营页不应设计聊天内容展示列');
console.log('管理员入口隐藏、角色显隐与运营后台隐私边界测试通过');
