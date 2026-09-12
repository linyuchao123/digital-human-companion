const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
const path = require('path');
const html = fs.readFileSync(path.join(__dirname, '..', 'integrated.html'), 'utf8');
let now = 0;
const values = {};
const ids = ['PARAM_ANGLE_Y', 'PARAM_BODY_Y','PARAM_ANGLE_X','PARAM_ARM_02_L_01','PARAM_ARM_02_L_02','PARAM_HAND_02_L'];
const parts={};
const core = {
  getParameterIndex: id => ids.indexOf(id),
  getParameterValueByIndex: index => values[ids[index]] || 0,
  setParameterValueByIndex: (index, value) => values[ids[index]] = value,
  setParameterValueById: (id, value) => values[id] = value,
  getPartIndex: id => ['PARTS_01_ARM_L_01','PARTS_01_ARM_L_02'].indexOf(id),
  setPartOpacityByIndex: (index,value) => parts[index]=value,
};
const ctx = {live2dModel: {internalModel: {coreModel: core}}, isPlayingMotion: false,
  Date: {now: () => 10000 + now}, performance: {now: () => now},
  wsParamsBuffer: {}, PARAM_MAP: {}, Math, Number, Object};
vm.createContext(ctx);
vm.runInContext(html.slice(html.indexOf('const GESTURES='), html.indexOf('let _blinkTimer=')), ctx);
const run = code => vm.runInContext(code, ctx);
assert.equal(run('gestureEnvelope(0)'), 0);
assert.equal(run('gestureEnvelope(1)'), 0);
assert.equal(run('gestureEnvelope(0.5)'), 1);
run("startGesture('Nod')");
let peak = 0;
for (now = 0; now < 1700; now += 20) {
  run('tickGesture(live2dModel.internalModel.coreModel)');
  peak = Math.max(peak, Math.abs(values.PARAM_ANGLE_Y));
  assert(Math.abs(values.PARAM_BODY_Y)<=0.5,'Body should barely move while nodding');
  const value = values.PARAM_ANGLE_Y;
  run('tickGesture(live2dModel.internalModel.coreModel)');
  assert.equal(values.PARAM_ANGLE_Y, value, 'Repeated frames must not accumulate offsets');
}
assert(peak > 10 && peak <= 12, 'Nod should be visible but bounded');
ctx.wsParamsBuffer.PARAM_ANGLE_Y = 3;
now = 2001;
run('tickGesture(live2dModel.internalModel.coreModel)');
assert.equal(values.PARAM_ANGLE_Y, 3, 'Return to current driver pose, not an outdated snapshot');
assert.equal(run('_gesture'), null);
assert(run("gestureWave('shake',0.25)") > 0);
assert(run("gestureWave('shake',0.75)") < 0);
assert.equal(run('wakePose(0).eye'),0);
assert.equal(run('wakePose(0.3).eye'),0);
assert(run('wakePose(0.9).eye')>0&&run('wakePose(0.9).eye')<1);
assert.equal(run('wakePose(1.5).eye'),1);
assert(run('wakePose(1.75).eye')<0.3,'Small sleepy blink after opening');
assert.equal(run('wakePose(2.0).eye'),1);
assert.equal(run('wakePose(2.55).release'),1);
for(let t=0;t<2.55;t+=0.01){
  const p=run(`wakePose(${t})`);
  assert(p.eye>=0&&p.eye<=1);assert(p.head>=-7&&p.head<=0);
}
assert(html.includes("if(motionName==='Hello') motionName='Idle'"));
console.log('Avatar gesture amplitude, no-drift and return-to-pose checks passed');
