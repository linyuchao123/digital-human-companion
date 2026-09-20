"""配置化社交授权和邮箱验证；所有密钥、验证码均仅在后端处理。"""
import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import smtplib
import sqlite3
import ssl
import time
from email.message import EmailMessage
from urllib.parse import urlencode, urlparse, parse_qs

import httpx
from fastapi import Request, BackgroundTasks
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from apps.api.security import allow_request


def init_auth_tables(conn):
    columns={row[1] for row in conn.execute('PRAGMA table_info(users)')}
    for name in ('email','email_verified_at'):
        if name not in columns:
            conn.execute(f"ALTER TABLE users ADD COLUMN {name} TEXT NOT NULL DEFAULT ''")
    if 'password_set' not in columns:
        conn.execute('ALTER TABLE users ADD COLUMN password_set INTEGER NOT NULL DEFAULT 1')
    token_columns={row[1] for row in conn.execute('PRAGMA table_info(auth_tokens)')}
    if 'source' not in token_columns: conn.execute("ALTER TABLE auth_tokens ADD COLUMN source TEXT NOT NULL DEFAULT 'password'")
    if 'issued_at' not in token_columns: conn.execute('ALTER TABLE auth_tokens ADD COLUMN issued_at REAL NOT NULL DEFAULT 0')
    conn.executescript("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_verified_email ON users(email) WHERE email<>'';
        CREATE TABLE IF NOT EXISTS social_accounts(provider TEXT,app_id TEXT,subject TEXT,user_id INTEGER,
            PRIMARY KEY(provider,app_id,subject),UNIQUE(provider,app_id,user_id));
        CREATE TABLE IF NOT EXISTS oauth_states(state_hash TEXT PRIMARY KEY,provider TEXT,browser_hash TEXT,
            expires REAL,user_id INTEGER,auth_token TEXT,ticket_hash TEXT);
        CREATE TABLE IF NOT EXISTS email_codes(user_id INTEGER PRIMARY KEY,email TEXT,salt TEXT,digest TEXT,
            expires REAL,attempts INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS password_reset_codes(user_id INTEGER PRIMARY KEY,email TEXT,salt TEXT,digest TEXT,
            expires REAL,attempts INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS registration_email_codes(email TEXT PRIMARY KEY,salt TEXT,digest TEXT,
            expires REAL,attempts INTEGER DEFAULT 0);
    """)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def configuration(provider):
    if provider not in ('wechat','qq'): return None
    prefix='WECHAT' if provider=='wechat' else 'QQ'
    app_id=os.getenv(prefix+'_APP_ID','').strip()
    secret=os.getenv(prefix+'_APP_SECRET','').strip()
    base=os.getenv('OAUTH_PUBLIC_BASE_URL','').rstrip('/')
    parsed=urlparse(base)
    valid=parsed.scheme=='https' or (os.getenv('PUBLIC_DEPLOYMENT','false')!='true' and parsed.scheme=='http' and parsed.hostname in ('localhost','127.0.0.1'))
    if not app_id or not secret or not valid or not parsed.hostname or parsed.query or parsed.fragment or parsed.username is not None or parsed.path not in ('','/'): return None
    return app_id,secret,base+'/api/auth/'+provider+'/callback'


def email_ready():
    return bool(os.getenv('SMTP_HOST') and os.getenv('SMTP_USERNAME') and os.getenv('SMTP_PASSWORD')
                and os.getenv('SMTP_FROM') and len(os.getenv('EMAIL_VERIFICATION_SECRET',''))>=32)


def normalize_email(value):
    value=value.strip().lower()
    if len(value)>254 or not re.fullmatch(r'[a-z0-9.!#$%&\x27*+/=?^_`{|}~-]+@[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.[a-z]{2,63}',value):
        raise ValueError('邮箱格式不正确')
    return value


def email_digest(user_id,email,salt,code):
    return hmac.new(os.environ['EMAIL_VERIFICATION_SECRET'].encode(),f'{user_id}:{email}:{salt}:{code}'.encode(),hashlib.sha256).hexdigest()


def registration_digest(email,salt,code):
    return hmac.new(os.environ['EMAIL_VERIFICATION_SECRET'].encode(),f'registration:{email}:{salt}:{code}'.encode(),hashlib.sha256).hexdigest()


def send_verification_email(address,code,purpose='邮箱绑定'):
    message=EmailMessage();message['From']=os.environ['SMTP_FROM'];message['To']=address
    message['Subject']=f'数字心屿 · {purpose}验证码'
    message.set_content(f'你的{purpose}验证码为：{code}\n10分钟内有效。如非本人操作，请忽略此邮件。\n请勿向任何人提供验证码。')
    context=ssl.create_default_context()
    mode=os.getenv('SMTP_SECURITY','ssl')
    port=int(os.getenv('SMTP_PORT','465' if mode=='ssl' else '587'))
    if mode=='ssl':
        client=smtplib.SMTP_SSL(os.environ['SMTP_HOST'],port,timeout=15,context=context)
    elif mode=='starttls':
        client=smtplib.SMTP(os.environ['SMTP_HOST'],port,timeout=15)
        client.ehlo();client.starttls(context=context);client.ehlo()
    else: raise ValueError('仅允许TLS邮件连接')
    with client:
        client.login(os.environ['SMTP_USERNAME'],os.environ['SMTP_PASSWORD']);client.send_message(message)


async def social_subject(provider,code):
    app_id,secret,callback=configuration(provider)
    async with httpx.AsyncClient(timeout=15,follow_redirects=False) as client:
        if provider=='wechat':
            result=await client.get('https://api.weixin.qq.com/sns/oauth2/access_token',params={
                'appid':app_id,'secret':secret,'code':code,'grant_type':'authorization_code'})
            result.raise_for_status();payload=result.json()
            if payload.get('errcode') or not payload.get('access_token'): raise ValueError('授权失败')
            subject=payload.get('openid')
        else:
            result=await client.get('https://graph.qq.com/oauth2.0/token',params={
                'grant_type':'authorization_code','client_id':app_id,'client_secret':secret,
                'code':code,'redirect_uri':callback,'fmt':'json'})
            result.raise_for_status()
            try: payload=result.json()
            except ValueError: payload={k:v[0] for k,v in parse_qs(result.text).items()}
            token=payload.get('access_token')
            if not token: raise ValueError('授权失败')
            result=await client.get('https://graph.qq.com/oauth2.0/me',params={'access_token':token,'fmt':'json'})
            result.raise_for_status()
            text=result.text.strip()
            match=re.fullmatch(r'callback\s*\((.*)\)\s*;?',text,re.S)
            payload=json.loads(match.group(1) if match else text)
            if str(payload.get('client_id'))!=app_id: raise ValueError('应用不匹配')
            subject=payload.get('openid')
    if not isinstance(subject,str) or not 1<=len(subject)<=256: raise ValueError('无效身份')
    return subject


class EmailRequest(BaseModel):
    email: str=Field(max_length=254)
    current_password: str=Field(default='',max_length=64)


class EmailVerify(EmailRequest):
    code: str=Field(pattern=r'^\d{6}$')


class SocialStart(BaseModel):
    bind: bool=False
    current_password: str=Field(default='',max_length=64)

class PasswordResetRequest(BaseModel):
    email: str=Field(min_length=3,max_length=254)

class PasswordResetConfirm(PasswordResetRequest):
    code: str=Field(pattern=r'^[0-9]{6}$')
    new_password: str=Field(min_length=8,max_length=64)


class RegistrationEmailRequest(BaseModel):
    email: str=Field(min_length=3,max_length=254)


class TicketExchange(BaseModel):
    ticket: str=Field(min_length=20,max_length=128)


def install_external_auth(app,get_db,verify,hash_password,check_password):
    def user(request): return verify(request.headers.get('X-Auth-Token',''))
    def error(message,status=400): return JSONResponse({'error':message},status_code=status)
    def token_result(conn,user_id):
        token=secrets.token_urlsafe(32)
        expires=time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime(time.time()+7*86400))
        conn.execute('INSERT INTO auth_tokens(token,user_id,expires_at,source,issued_at) VALUES(?,?,?,?,?)',(token,user_id,expires,'oauth',time.time()))
        username=conn.execute('SELECT username FROM users WHERE id=?',(user_id,)).fetchone()[0]
        return {'token':token,'username':username,'user_id':user_id}

    @app.get('/api/auth/options')
    async def options():
        return {'wechat':bool(configuration('wechat')),'qq':bool(configuration('qq')),'email_binding':email_ready(),
                'password_recovery':email_ready(),'email_registration':email_ready()}

    @app.post('/api/auth/register/email/send')
    async def send_registration_email(payload: RegistrationEmailRequest,request: Request):
        if not email_ready():return error('邮件服务尚未配置，暂不能使用邮箱验证码注册',503)
        try:address=normalize_email(payload.email)
        except ValueError:return error('邮箱格式不正确')
        ip=request.client.host if request.client else 'unknown'
        if not allow_request(('register-email-ip',ip),5,900) or not allow_request(('register-email-address',digest(address)),1,60):
            return error('发送过于频繁，请稍后再试',429)
        with get_db() as conn:
            conn.execute('DELETE FROM registration_email_codes WHERE expires<?',(time.time(),))
            if conn.execute("SELECT id FROM users WHERE email=? AND email_verified_at<>''",(address,)).fetchone():
                return error('该邮箱已注册，可以直接登录或找回密码',409)
            code=f'{secrets.randbelow(1000000):06d}';salt=secrets.token_hex(16)
            conn.execute('INSERT OR REPLACE INTO registration_email_codes VALUES(?,?,?,?,0)',
                         (address,salt,registration_digest(address,salt,code),time.time()+600));conn.commit()
        try:await asyncio.to_thread(send_verification_email,address,code,'注册')
        except Exception:
            with get_db() as conn:
                conn.execute('DELETE FROM registration_email_codes WHERE email=? AND salt=?',(address,salt));conn.commit()
            return error('邮件发送失败，请检查邮件服务配置',502)
        return {'ok':True,'expires_in':600,'message':'验证码已发送，10分钟内有效'}

    def reset_digest(user_id,address,salt,code):
        return email_digest(user_id,address,'password-reset:'+salt,code)

    def deliver_reset(user_id,address,salt,code):
        try:
            send_verification_email(address,code,'密码重置')
        except Exception:
            # 不记录邮件、验证码或异常正文；迟到失败不能删除新验证码。
            with get_db() as conn:
                conn.execute('DELETE FROM password_reset_codes WHERE user_id=? AND salt=?',(user_id,salt))
                conn.commit()

    @app.post('/api/auth/password/reset/send')
    async def send_reset(payload: PasswordResetRequest,request: Request,background: BackgroundTasks):
        if not email_ready():return error('邮件服务尚未配置，暂不能通过邮箱找回密码',503)
        try:address=normalize_email(payload.email)
        except ValueError:return error('邮箱格式不正确')
        ip=request.client.host if request.client else 'unknown'
        if not allow_request(('reset-ip',ip),5,900) or not allow_request(('reset-address',digest(address)),1,60):
            return error('发送过于频繁，请稍后再试',429)
        with get_db() as conn:
            row=conn.execute("SELECT id FROM users WHERE email=? AND email_verified_at<>''",(address,)).fetchone()
            if row:
                code=f'{secrets.randbelow(1000000):06d}';salt=secrets.token_hex(16)
                conn.execute('INSERT OR REPLACE INTO password_reset_codes VALUES(?,?,?,?,?,0)',
                             (row['id'],address,salt,reset_digest(row['id'],address,salt,code),time.time()+600))
                conn.commit()
                background.add_task(deliver_reset,row['id'],address,salt,code)
        return {'message':'如果该邮箱已绑定并验证，我们会发送重置验证码。请检查收件箱及垃圾邮件；60秒后可重试。'}

    @app.post('/api/auth/password/reset/confirm')
    async def confirm_reset(payload: PasswordResetConfirm,request: Request):
        if not email_ready():return error('邮件服务尚未配置',503)
        try:address=normalize_email(payload.email)
        except ValueError:return error('邮箱格式不正确')
        if len(payload.new_password.encode())>72:return error('密码UTF-8长度不超过72字节')
        ip=request.client.host if request.client else 'unknown'
        if not allow_request(('reset-confirm',ip),20,900):return error('验证过于频繁，请稍后再试',429)
        with get_db() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row=conn.execute("SELECT r.* FROM password_reset_codes r JOIN users u ON u.id=r.user_id "
                             "WHERE r.email=? AND u.email=r.email AND u.email_verified_at<>''",(address,)).fetchone()
            if not row or row['expires']<=time.time() or row['attempts']>=5:
                return error('验证码不正确或已失效，请重新发送')
            conn.execute('UPDATE password_reset_codes SET attempts=attempts+1 WHERE user_id=?',(row['user_id'],))
            if not hmac.compare_digest(row['digest'],reset_digest(row['user_id'],address,row['salt'],payload.code)):
                conn.commit();return error('验证码不正确或已失效，请重新发送')
            conn.execute('UPDATE users SET password_hash=?,password_set=1 WHERE id=?',
                         (hash_password(payload.new_password),row['user_id']))
            conn.execute('DELETE FROM auth_tokens WHERE user_id=?',(row['user_id'],))
            conn.execute('DELETE FROM oauth_states WHERE user_id=?',(row['user_id'],))
            conn.execute('DELETE FROM email_codes WHERE user_id=?',(row['user_id'],))
            conn.execute('DELETE FROM password_reset_codes WHERE user_id=?',(row['user_id'],))
            conn.commit()
        return {'message':'密码已重置，旧登录已失效。请使用新密码登录。'}

    @app.post('/api/auth/{provider}/start')
    async def start(provider: str,payload: SocialStart,request: Request):
        config=configuration(provider)
        if not config: return error('此登录方式尚未配置应用凭证与回调地址',503)
        user_id=user(request) if payload.bind else None
        if payload.bind:
            if user_id is None: return error('请先登录',401)
            with get_db() as conn:
                row=conn.execute('SELECT password_hash FROM users WHERE id=?',(user_id,)).fetchone()
            if not check_password(payload.current_password,row[0]): return error('绑定第三方身份需要当前密码',403)
        nonce=secrets.token_urlsafe(32);browser=secrets.token_urlsafe(32)
        with get_db() as conn:
            conn.execute('DELETE FROM oauth_states WHERE expires<?',(time.time(),))
            conn.execute('INSERT INTO oauth_states VALUES(?,?,?,?,?,?,NULL)',(digest(nonce),provider,digest(browser),time.time()+600,user_id,request.headers.get('X-Auth-Token','') if user_id else ''))
            conn.commit()
        app_id,_,callback=config
        params={'redirect_uri':callback,'response_type':'code','state':nonce}
        if provider=='wechat':
            params.update(appid=app_id,scope='snsapi_login');url='https://open.weixin.qq.com/connect/qrconnect'
        else:
            params.update(client_id=app_id,scope='get_user_info');url='https://graph.qq.com/oauth2.0/authorize'
        response=JSONResponse({'url':url+'?'+urlencode(params)})
        response.set_cookie('oauth_browser',browser,httponly=True,secure=callback.startswith('https:'),samesite='lax',max_age=600,path='/api/auth')
        return response

    @app.get('/api/auth/{provider}/callback')
    async def callback(provider: str,request: Request,code: str='',state: str=''):
        if not configuration(provider): return error('授权尚未配置',503)
        with get_db() as conn:
            row=conn.execute('SELECT * FROM oauth_states WHERE state_hash=?',(digest(state),)).fetchone()
            if not row or row['provider']!=provider or row['expires']<time.time() or row['ticket_hash'] or not hmac.compare_digest(row['browser_hash'],digest(request.cookies.get('oauth_browser',''))):
                return error('授权已过期或状态不匹配，请重新登录')
            conn.execute('DELETE FROM oauth_states WHERE state_hash=?',(digest(state),));conn.commit()
        if not code or len(code)>512: return RedirectResponse('/#oauth_error=cancelled',status_code=303)
        try: subject=await social_subject(provider,code)
        except Exception: return RedirectResponse('/#oauth_error=failed',status_code=303)
        app_id=configuration(provider)[0]
        with get_db() as conn:
            existing=conn.execute('SELECT user_id FROM social_accounts WHERE provider=? AND app_id=? AND subject=?',(provider,app_id,subject)).fetchone()
            user_id=row['user_id']
            if user_id is not None:
                if verify(row['auth_token'])!=user_id: return error('登录已失效，请重新绑定',401)
                if existing and existing[0]!=user_id: return error('该第三方身份已绑定其他账号',409)
            elif existing: user_id=existing[0]
            else:
                username=provider+'_'+secrets.token_hex(5)
                cursor=conn.execute('INSERT INTO users(username,password_hash,created_at,display_name,password_set) VALUES(?,?,?,?,0)',
                    (username,hash_password(secrets.token_urlsafe(32)),time.strftime('%Y-%m-%dT%H:%M:%S'),'微信用户' if provider=='wechat' else 'QQ用户'))
                user_id=cursor.lastrowid
            if not existing:
                try: conn.execute('INSERT INTO social_accounts VALUES(?,?,?,?)',(provider,app_id,subject,user_id))
                except sqlite3.IntegrityError:
                    conn.rollback();return error('该账号已有此平台绑定或授权同时进行，请重试',409)
            ticket=secrets.token_urlsafe(32)
            conn.execute('INSERT INTO oauth_states VALUES(?,?,?,?,?,?,?)',(digest(ticket),provider,row['browser_hash'],time.time()+60,user_id,'',digest(ticket)))
            conn.commit()
        return RedirectResponse('/#oauth_ticket='+ticket,status_code=303)

    @app.post('/api/auth/exchange')
    async def exchange(payload: TicketExchange,request: Request):
        with get_db() as conn:
            row=conn.execute('SELECT * FROM oauth_states WHERE state_hash=?',(digest(payload.ticket),)).fetchone()
            if not row or row['ticket_hash']!=digest(payload.ticket) or row['expires']<time.time() or not hmac.compare_digest(row['browser_hash'],digest(request.cookies.get('oauth_browser',''))): return error('登录票据无效或已过期')
            conn.execute('DELETE FROM oauth_states WHERE state_hash=?',(digest(payload.ticket),))
            result=token_result(conn,row['user_id']);conn.commit()
        response=JSONResponse(result);response.delete_cookie('oauth_browser',path='/api/auth');return response

    @app.post('/api/profile/email/send')
    async def send_email(payload: EmailRequest,request: Request):
        user_id=user(request)
        if user_id is None: return error('请先登录',401)
        if not email_ready(): return error('邮件服务尚未配置',503)
        try: address=normalize_email(payload.email)
        except ValueError: return error('邮箱格式不正确')
        with get_db() as conn:
            account=conn.execute('SELECT password_hash,password_set FROM users WHERE id=?',(user_id,)).fetchone()
            session=conn.execute('SELECT source,issued_at FROM auth_tokens WHERE token=?',(request.headers.get('X-Auth-Token',''),)).fetchone()
        fresh_social=not account['password_set'] and session and session['source']=='oauth' and session['issued_at']>time.time()-600
        if not fresh_social and not check_password(payload.current_password,account['password_hash']): return error('绑定邮箱需验证当前密码；社交账号请重新授权后操作',403)
        if not allow_request(('email-user',user_id),1) or not allow_request(('email-address',address),3,900): return error('发送过于频繁，请稍后再试',429)
        code=f'{secrets.randbelow(1000000):06d}';salt=secrets.token_hex(16)
        with get_db() as conn:
            if conn.execute('SELECT id FROM users WHERE email=? AND id<>?',(address,user_id)).fetchone(): return error('该邮箱暂不能绑定',409)
            conn.execute('INSERT OR REPLACE INTO email_codes VALUES(?,?,?,?,?,0)',(user_id,address,salt,email_digest(user_id,address,salt,code),time.time()+600));conn.commit()
        try: await asyncio.to_thread(send_verification_email,address,code)
        except Exception:
            with get_db() as conn: conn.execute('DELETE FROM email_codes WHERE user_id=? AND salt=?',(user_id,salt));conn.commit()
            return error('邮件发送失败，请检查邮件服务配置',502)
        return {'ok':True,'expires_in':600}

    @app.post('/api/profile/email/verify')
    async def verify_email(payload: EmailVerify,request: Request):
        user_id=user(request)
        if user_id is None: return error('请先登录',401)
        if not email_ready(): return error('邮件服务尚未配置',503)
        try: address=normalize_email(payload.email)
        except ValueError: return error('邮箱格式不正确')
        with get_db() as conn:
            row=conn.execute('SELECT * FROM email_codes WHERE user_id=?',(user_id,)).fetchone()
            if not row or row['expires']<time.time() or row['attempts']>=5: return error('验证码已失效，请重新发送')
            conn.execute('UPDATE email_codes SET attempts=attempts+1 WHERE user_id=?',(user_id,));conn.commit()
            if row['email']!=address or not hmac.compare_digest(row['digest'],email_digest(user_id,address,row['salt'],payload.code)): return error('验证码不正确')
            try:
                conn.execute('UPDATE users SET email=?,email_verified_at=? WHERE id=?',(address,time.strftime('%Y-%m-%dT%H:%M:%S'),user_id))
                conn.execute('DELETE FROM email_codes WHERE user_id=?',(user_id,));conn.commit()
            except sqlite3.IntegrityError: return error('该邮箱暂不能绑定',409)
        return {'ok':True}
