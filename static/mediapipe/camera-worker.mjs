/* Classic Worker is required by this release's WASM importScripts loader. */
const exports = {};
importScripts('./tasks-vision/vision_bundle.js');
const {FaceLandmarker, FilesetResolver} = exports;
let detector;
const qualityCanvas = new OffscreenCanvas(32,24);
const qualityContext = qualityCanvas.getContext('2d',{willReadFrequently:true});
self.onmessage = async ({data}) => {
  if (data.type === 'init') {
    try {
      detector = await FaceLandmarker.createFromOptions(await FilesetResolver.forVisionTasks('/static/mediapipe/tasks-vision/wasm'), {
        baseOptions: {modelAssetPath: '/api/vision/face-landmarker-model', delegate: 'CPU'},
        runningMode: 'VIDEO', numFaces: 2, outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
        minFaceDetectionConfidence: .6, minFacePresenceConfidence: .6, minTrackingConfidence: .6,
      });
      self.postMessage({type: 'ready'});
    } catch(error) { self.postMessage({type: 'error', stage: 'init', detail: String(error.message || '初始化失败').slice(0,300)}); }
    return;
  }
  if (data.type !== 'frame') return;
  const started = performance.now();
  try {
    qualityContext.drawImage(data.bitmap,0,0,32,24);
    const pixels = qualityContext.getImageData(0,0,32,24).data;
    let brightness=0;
    for(let i=0;i<pixels.length;i+=4) brightness+=(pixels[i]+pixels[i+1]+pixels[i+2])/3;
    brightness/=pixels.length/4;
    const result = detector.detectForVideo(data.bitmap, data.timestamp);
    const count = result.faceLandmarks.length;
    const scores = Object.fromEntries((result.faceBlendshapes[0]?.categories || []).map(c => [c.categoryName, c.score]));
    const avg = (a,b) => ((scores[a] || 0)+(scores[b] || 0))/2;
    const matrix = result.facialTransformationMatrixes[0]?.data;
    const deg = 180/Math.PI;
    const features = {smile: avg('mouthSmileLeft','mouthSmileRight'), brow_down: avg('browDownLeft','browDownRight'),
      eye_closed: avg('eyeBlinkLeft','eyeBlinkRight'), yaw: matrix ? Math.asin(Math.max(-1,Math.min(1,-matrix[2])))*deg : 0,
      pitch: matrix ? Math.atan2(matrix[6],matrix[10])*deg : 0,
      roll: matrix ? Math.atan2(matrix[1],matrix[0])*deg : 0};
    const points = result.faceLandmarks[0] || [];
    const width = points.length ? Math.max(...points.map(p=>p.x))-Math.min(...points.map(p=>p.x)) : 0;
    const inside = points.length && points.every(p=>p.x>=0 && p.x<=1 && p.y>=0 && p.y<=1);
    self.postMessage({type: 'result', face_count: count, quality: width > .16 && inside && matrix && brightness>35 && brightness<235 ? 'good' : 'poor',
      features, duration: performance.now()-started});
  } catch(error) { self.postMessage({type: 'error', stage: 'frame', detail: String(error.message || '推理失败').slice(0,300)}); }
  finally { data.bitmap.close(); }
};
