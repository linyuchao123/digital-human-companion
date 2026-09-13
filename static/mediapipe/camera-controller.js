/* Raw camera frames stay in this browser and its worker. No frame uploads. */
(() => {
  let generation=0, stream=null, worker=null, timer=null, busy=false, lastSent=0, seq=0, id='', interval=125, timeout=null, starting=false;
  const video=()=>document.getElementById('camera-video');
  const status=text=>{document.getElementById('camera-status').textContent=text;};
  window.stopCameraPerception = (reason='已关闭') => {
    generation++;
    clearTimeout(timer); clearTimeout(timeout);
    if(id) chatSend({type:'vision_control',enabled:false,stream_id:id});
    id=''; worker?.terminate(); worker=null;
    stream?.getTracks().forEach(t=>t.stop()); stream=null;
    video().srcObject=null; busy=false; starting=false;
    document.getElementById('camera-btn').classList.remove('on'); status(reason);
  };
  window.toggleCameraPreview=()=>{video().hidden=!video().hidden;};
  window.toggleCameraPerception=async()=>{
    if(worker || stream || starting){stopCameraPerception();return;}
    if(!chatWsReady){status('请先连接数字人');return;}
    if(!navigator.mediaDevices?.getUserMedia){status('请使用 localhost 或 HTTPS 打开页面');return;}
    if(!confirm('摄像头画面仅在本机处理，不上传、不录制、不保存。数值特征发往项目后端；简短观察描述可能随对话发送给当前大模型。是否开启？')) return;
    const epoch=++generation; starting=true; status('加载中');
    document.getElementById('camera-panel').hidden=false;
    document.getElementById('camera-btn').classList.add('on');
    try {
      const acquired=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},frameRate:{ideal:15,max:15}},audio:false});
      if(epoch!==generation){acquired.getTracks().forEach(t=>t.stop());return;}
      stream=acquired; stream.getVideoTracks()[0].onended=()=>stopCameraPerception('摄像头已停止');
      video().srcObject=stream; await video().play();
      if(epoch!==generation) return;
      id=crypto.randomUUID(); seq=0; lastSent=0; interval=125;
      if(!chatSend({type:'vision_control',enabled:true,stream_id:id})) throw Error('connection');
      worker=new Worker('/static/mediapipe/camera-worker.mjs',{type:'module'});
      timeout=setTimeout(()=>stopCameraPerception('视觉模型加载超时'),20000);
      const pump=async()=>{
        if(epoch!==generation || !worker) return;
        if(!document.hidden && !busy && video().readyState>=2){
          busy=true;
          try {
            const bitmap=await createImageBitmap(video());
            if(epoch!==generation || !worker){bitmap.close();return;}
            worker.postMessage({type:'frame',bitmap,timestamp:performance.now()},[bitmap]);
          } catch { stopCameraPerception('摄像头处理失败');return; }
        }
        timer=setTimeout(pump,interval);
      };
      worker.onerror=()=>stopCameraPerception('视觉模型加载失败，请检查本地资源');
      worker.onmessage=({data})=>{
        if(epoch!==generation) return;
        if(data.type==='ready'){clearTimeout(timeout);starting=false;status('观察中');pump();return;}
        if(data.type==='error'){stopCameraPerception('视觉模型处理失败');return;}
        busy=false;
        if(data.duration>125) interval=250;
        status(data.face_count===0?'未检测到人脸':data.face_count>1?'多人入镜':data.quality==='poor'?'画面不清晰':'观察中');
        if(performance.now()-lastSent>=520){
          lastSent=performance.now();
          if(!chatSend({type:'vision_features',stream_id:id,seq:seq++,face_count:data.face_count,quality:data.quality,features:data.features})) stopCameraPerception('连接已断开');
        }
      };
      worker.postMessage({type:'init'});
    } catch(error) {
      if(epoch===generation) stopCameraPerception(error.name==='NotAllowedError'?'摄像头权限被拒绝':error.name==='NotFoundError'?'未找到摄像头':'摄像头无法开启，请检查设备占用');
    }
  };
  document.addEventListener('visibilitychange',()=>{if(document.hidden && (stream || worker || starting)) stopCameraPerception('页面已隐藏，摄像头已关闭');});
  window.addEventListener('pagehide',()=>stopCameraPerception());
})();
