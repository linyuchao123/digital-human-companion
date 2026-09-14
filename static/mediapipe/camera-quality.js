/* Local-only quality hints; these are heuristics, never emotion confidence. */
self.cameraQuality = (points, matrix, brightness, count, hasExpressions) => {
  if(count===0) return {quality:'poor',reason:'未检测到人脸，请面向摄像头'};
  if(count>1) return {quality:'poor',reason:'多人入镜，暂停个人状态观察'};
  if(!points.length || points.some(p=>!Number.isFinite(p.x)||!Number.isFinite(p.y)))
    return {quality:'poor',reason:'面部特征不稳定，请保持片刻'};
  const xs=points.map(p=>p.x),ys=points.map(p=>p.y);
  const width=Math.max(...xs)-Math.min(...xs);
  if(width<.12) return {quality:'poor',reason:'距离较远，请稍微靠近摄像头'};
  const inside=points.filter(p=>p.x>=-.02&&p.x<=1.02&&p.y>=-.02&&p.y<=1.02).length/points.length;
  if(inside<.95) return {quality:'poor',reason:'面部未完整入镜，请调整位置'};
  if(brightness<30) return {quality:'poor',reason:'面部光线偏暗，请增加照明'};
  if(brightness>245) return {quality:'poor',reason:'面部光线过亮，请避开强光'};
  if(!matrix || matrix.length!==16 || Array.from(matrix).some(x=>!Number.isFinite(x)) || !hasExpressions)
    return {quality:'poor',reason:'表情特征尚未就绪，请保持片刻'};
  return {quality:'good',reason:'观察中 · 已检测到单人面部'};
};
