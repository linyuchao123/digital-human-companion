import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from apps.api import integrated_server as server
from apps.api.security import _budgets, allow_request


class AccountSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.original=server.DB_PATH
        server.DB_PATH=Path(self.temp.name)/'users.db'
        server._init_db();_budgets.clear()
        self.client=TestClient(server.app)

    def tearDown(self):
        self.client.close();server.DB_PATH=self.original;self.temp.cleanup();_budgets.clear()

    def register(self,name='profile-user'):
        result=self.client.post('/api/auth/register',json={'username':name,'password':'test-password'})
        self.assertEqual(result.status_code,200)
        return {'X-Auth-Token':result.json()['token']}

    def test_profile_private_and_validated(self):
        self.assertEqual(self.client.get('/api/profile').status_code,401)
        headers=self.register()
        payload={'username':'profile-user','display_name':'小林','birthday':'2000-01-01'}
        self.assertEqual(self.client.patch('/api/profile',headers=headers,json=payload).status_code,200)
        profile=self.client.get('/api/profile',headers=headers).json()
        self.assertEqual(profile['display_name'],'小林');self.assertNotIn('password_hash',profile)
        other=self.register('other-profile')
        self.assertEqual(self.client.get('/api/profile',headers=other).json()['display_name'],'')
        payload['birthday']='2999-01-01'
        self.assertEqual(self.client.patch('/api/profile',headers=headers,json=payload).status_code,400)
        payload['birthday']='';payload['avatar']='data:image/svg+xml;base64,AAA'
        self.assertEqual(self.client.patch('/api/profile',headers=headers,json=payload).status_code,400)

    def test_password_change_revokes_tokens_and_username_needs_password(self):
        headers=self.register()
        payload={'username':'new-profile'}
        self.assertEqual(self.client.patch('/api/profile',headers=headers,json=payload).status_code,403)
        payload['current_password']='test-password'
        self.assertEqual(self.client.patch('/api/profile',headers=headers,json=payload).status_code,200)
        result=self.client.post('/api/profile/password',headers=headers,json={'current_password':'test-password','new_password':'new-password'})
        self.assertEqual(result.status_code,200)
        self.assertEqual(self.client.get('/api/profile',headers=headers).status_code,401)
        self.assertEqual(self.client.post('/api/auth/login',json={'username':'new-profile','password':'test-password'}).status_code,401)
        self.assertEqual(self.client.post('/api/auth/login',json={'username':'new-profile','password':'new-password'}).status_code,200)

    def test_missing_bcrypt_fails_closed(self):
        with patch.object(server,'HAS_BCRYPT',False):
            self.assertEqual(self.client.post('/api/auth/register',json={'username':'no-bcrypt','password':'test-password'}).status_code,503)
            with self.assertRaises(RuntimeError): server._hash_password('secret')

    def test_public_costly_endpoints_and_upload_size(self):
        with patch.dict('os.environ',{'PUBLIC_DEPLOYMENT':'true'}):
            self.assertEqual(self.client.post('/api/tts',json={'text':'你好'}).status_code,401)
            self.assertEqual(self.client.post('/api/upload_video',files={'file':('test.mp4',b'x')}).status_code,401)
        self.assertEqual(self.client.post('/api/auth/login',content=b'x'*262145).status_code,413)
        self.assertTrue(allow_request('small-budget',1));self.assertFalse(allow_request('small-budget',1))

    def test_driver_does_not_broadcast_to_other_users(self):
        import asyncio
        class Socket:
            async def send_json(self,message): raise AssertionError('Cross-session broadcast')
        observer=Socket();server._drive_clients.add(observer)
        async def scenario():
            with patch.object(server,'_compute_live2d_params',return_value={'ParamAngleX':7}),patch.object(server,'_send') as send:
                task=asyncio.create_task(server._drive_loop(object(),server.SessionState('isolated'),asyncio.get_running_loop()))
                await asyncio.sleep(0.08);task.cancel()
                await asyncio.gather(task,return_exceptions=True)
                self.assertTrue(send.called)
        try: asyncio.run(scenario())
        finally: server._drive_clients.discard(observer)
