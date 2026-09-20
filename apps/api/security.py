"""单进程开发/演示部署的请求预算；多副本应使用共享限流存储。"""
import time
from collections import deque

from fastapi.responses import JSONResponse

_budgets = {}


def allow_request(key, limit, window=60):
    now = time.monotonic()
    if len(_budgets) > 5000:
        for existing in list(_budgets):
            if not _budgets[existing] or _budgets[existing][-1] < now-window:
                del _budgets[existing]
        if key not in _budgets and len(_budgets) > 5000:
            return False
    queue = _budgets.setdefault(key, deque())
    while queue and queue[0] <= now-window:
        queue.popleft()
    if len(queue) >= limit:
        return False
    queue.append(now)
    return True


def install_http_security(app, verify, public):
    @app.middleware("http")
    async def security(request, call_next):
        path=request.url.path
        costly=path in {"/api/tts", "/api/tts/stream", "/api/agent/chat", "/api/upload_video",
                        "/api/avatar/livetalking/offer", "/api/avatar/livetalking/audio"}
        protected=costly or path.startswith('/api/avatar/livetalking/')
        user=verify(request.headers.get('X-Auth-Token',''))
        if protected and public() and user is None:
            return JSONResponse({'error':'请先登录'},status_code=401)
        if path.startswith('/api/auth/') or costly or path.startswith('/api/profile'):
            key=(request.client.host if request.client else 'unknown',user,'auth' if '/auth/' in path else 'costly')
            if not allow_request(key,(20 if public() else 120) if '/auth/' in path else 30):
                return JSONResponse({'error':'请求过于频繁，请稍后重试'},status_code=429,headers={'Retry-After':'60'})
        if request.method in {'POST','PATCH','PUT'}:
            maximum=20*1024*1024 if path in {'/api/upload_video','/api/avatar/livetalking/audio'} else 256*1024
            body=bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body)>maximum:
                    return JSONResponse({'error':'请求内容过大'},status_code=413)
            request._body=bytes(body)
        response=await call_next(request)
        if path in {'/static/mediapipe/camera-controller.js', '/static/mediapipe/camera-worker.mjs', '/static/mediapipe/camera-quality.js'}:
            response.headers['Cache-Control']='no-store'
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='same-origin'
        if path.startswith('/api/'):
            response.headers['Cache-Control']='no-store'
        return response
