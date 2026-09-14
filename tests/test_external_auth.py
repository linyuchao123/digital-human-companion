import os
from urllib.parse import urlparse, parse_qs
from unittest.mock import AsyncMock, patch

from tests.test_account_security import AccountSecurityTests
from apps.api import external_auth as auth, integrated_server as server


class ExternalAuthTests(AccountSecurityTests):
    def setUp(self):
        super().setUp()
        self.env = patch.dict(os.environ, {
            'PUBLIC_DEPLOYMENT': 'false', 'OAUTH_PUBLIC_BASE_URL': 'http://127.0.0.1:8801',
            'QQ_APP_ID': 'test-app', 'QQ_APP_SECRET': 'test-secret',
            'SMTP_HOST': 'mail.example.org', 'SMTP_USERNAME': 'test',
            'SMTP_PASSWORD': 'test', 'SMTP_FROM': 'test@example.org',
            'EMAIL_VERIFICATION_SECRET': 'x' * 32,
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()
        super().tearDown()

    def test_oauth_browser_state_and_one_time_ticket(self):
        start = self.client.post('/api/auth/qq/start', json={})
        state = parse_qs(urlparse(start.json()['url']).query)['state'][0]
        with patch.object(auth, 'social_subject', new=AsyncMock(return_value='test-subject')) as exchange:
            bad = self.client.get('/api/auth/qq/callback', params={'state': 'bad', 'code': 'test'}, follow_redirects=False)
            self.assertEqual(bad.status_code, 400)
            exchange.assert_not_called()
            result = self.client.get('/api/auth/qq/callback', params={'state': state, 'code': 'test'}, follow_redirects=False)
            self.assertEqual(result.status_code, 303)
            ticket = parse_qs(urlparse(result.headers['location']).fragment)['oauth_ticket'][0]
            self.assertEqual(self.client.get('/api/auth/qq/callback', params={'state': state, 'code': 'test'}).status_code, 400)
        login = self.client.post('/api/auth/exchange', json={'ticket': ticket})
        self.assertEqual(login.status_code, 200)
        self.assertEqual(self.client.post('/api/auth/exchange', json={'ticket': ticket}).status_code, 400)
        profile = self.client.get('/api/profile', headers={'X-Auth-Token': login.json()['token']}).json()
        self.assertEqual(profile['password_set'], 0)

    def test_verified_email_login_and_replay(self):
        headers = self.register()
        payload = {'email': 'Person@example.org', 'current_password': 'test-password'}
        with patch.object(auth, 'send_verification_email') as sender:
            self.assertEqual(self.client.post('/api/profile/email/send', headers=headers, json={'email': payload['email']}).status_code, 403)
            sent = self.client.post('/api/profile/email/send', headers=headers, json=payload)
            self.assertEqual(sent.status_code, 200)
            code = sender.call_args.args[1]
            self.assertNotIn(code, sent.text)
        payload['code'] = code
        self.assertEqual(self.client.post('/api/profile/email/verify', headers=headers, json=payload).status_code, 200)
        self.assertEqual(self.client.post('/api/profile/email/verify', headers=headers, json=payload).status_code, 400)
        login = self.client.post('/api/auth/login', json={'username': 'PERSON@example.org', 'password': 'test-password'})
        self.assertEqual(login.status_code, 200)
        self.assertEqual(login.json()['username'], 'profile-user')

    def test_callback_requires_same_browser_and_binding_password(self):
        headers = self.register()
        self.assertEqual(self.client.post('/api/auth/qq/start', headers=headers, json={'bind': True}).status_code, 403)
        started = self.client.post('/api/auth/qq/start', headers=headers, json={'bind': True, 'current_password': 'test-password'})
        state = parse_qs(urlparse(started.json()['url']).query)['state'][0]
        self.client.cookies.clear()
        with patch.object(auth, 'social_subject', new=AsyncMock()) as exchange:
            self.assertEqual(self.client.get('/api/auth/qq/callback', params={'state': state, 'code': 'test'}).status_code, 400)
            exchange.assert_not_called()

    def test_email_attempt_limit_and_expiry(self):
        headers = self.register()
        payload = {'email': 'person@example.org', 'current_password': 'test-password'}
        with patch.object(auth, 'send_verification_email') as sender:
            self.client.post('/api/profile/email/send', headers=headers, json=payload)
            code = sender.call_args.args[1]
        wrong = '000000' if code != '000000' else '111111'
        for _ in range(5):
            self.assertEqual(self.client.post('/api/profile/email/verify', headers=headers, json={**payload, 'code': wrong}).status_code, 400)
        self.assertEqual(self.client.post('/api/profile/email/verify', headers=headers, json={**payload, 'code': code}).status_code, 400)
        with server._get_db() as conn:
            conn.execute('UPDATE email_codes SET attempts=0, expires=0'); conn.commit()
        self.assertEqual(self.client.post('/api/profile/email/verify', headers=headers, json={**payload, 'code': code}).status_code, 400)
