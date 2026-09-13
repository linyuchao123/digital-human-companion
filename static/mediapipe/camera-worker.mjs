/* Classic Worker is required by this release's WASM importScripts loader. */
const exports = {};
importScripts('./tasks-vision/vision_bundle.js');
importScripts('./camera-quality.js?v=20260914-3');
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
    // Measure the face region, not the whole room (dark backgrounds are common).
    let brightness=0, samples=0;
    const xs=points.map(p=>p.x),ys=points.map(p=>p.y);
    const left=Math.max(0,Math.floor(Math.min(...xs)*32)),right=Math.min(32,Math.ceil(Math.max(...xs)*32));
    const top=Math.max(0,Math.floor(Math.min(...ys)*24)),bottom=Math.min(24,Math.ceil(Math.max(...ys)*24));
    for(let y=top;y<bottom;y++) for(let x=left;x<right;x++) {
      const i=(y*32+x)*4;brightness+=(pixels[i]+pixels[i+1]+pixels[i+2])/3;samples++;
    }
    brightness=samples?brightness/samples:0;
    const assessment=self.cameraQuality(points,matrix,brightness,count,Object.keys(scores).length>0);
    self.postMessage({type: 'result', face_count: count, ...assessment,
      features, duration: performance.now()-started});
  } catch(error) { self.postMessage({type: 'error', stage: 'frame', detail: String(error.message || '推理失败').slice(0,300)}); }
  finally { data.bitmap.close(); }
};
