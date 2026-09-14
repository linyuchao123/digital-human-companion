import os
from unittest.mock import patch
from tests.test_account_security import AccountSecurityTests
from apps.api import external_auth as auth, integrated_server as server
from apps.api.security import _budgets


class PasswordRecoveryTests(AccountSecurityTests):
    def setUp(self):
        super().setUp()
        self.env=patch.dict(os.environ,{'SMTP_HOST':'mail.example.org','SMTP_USERNAME':'test',
            'SMTP_PASSWORD':'test','SMTP_FROM':'test@example.org','EMAIL_VERIFICATION_SECRET':'x'*32})
        self.env.start()

    def tearDown(self):
        self.env.stop();super().tearDown()

    def prepare(self):
        headers=self.register()
        with server._get_db() as conn:
            conn.execute("UPDATE users SET email='person@example.org',email_verified_at='2026' WHERE username='profile-user'")
            conn.commit()
        with patch.object(auth,'send_verification_email') as sender:
            result=self.client.post('/api/auth/password/reset/send',json={'email':'Person@example.org'})
            self.assertEqual(result.status_code,200)
            code=sender.call_args.args[1]
            self.assertEqual(sender.call_args.args[2],'密码重置')
            self.assertNotIn(code,result.text)
        return headers,{'email':'person@example.org','code':code,'new_password':'new-password'}

    def test_reset_one_time_revokes_login_and_hashes_code(self):
        headers,payload=self.prepare()
        with server._get_db() as conn:
            row=conn.execute('SELECT * FROM password_reset_codes').fetchone()
            self.assertNotEqual(row['digest'],payload['code'])
        result=self.client.post('/api/auth/password/reset/confirm',json=payload)
        self.assertEqual(result.status_code,200)
        self.assertEqual(self.client.get('/api/profile',headers=headers).status_code,401)
        self.assertEqual(self.client.post('/api/auth/password/reset/confirm',json=payload).status_code,400)
        self.assertEqual(self.client.post('/api/auth/login',json={'username':'profile-user','password':'test-password'}).status_code,401)
        self.assertEqual(self.client.post('/api/auth/login',json={'username':'person@example.org','password':'new-password'}).status_code,200)

    def test_unknown_email_same_response_and_no_send(self):
        _,_=self.prepare()
        _budgets.clear()
        with patch.object(auth,'send_verification_email') as sender:
            unknown=self.client.post('/api/auth/password/reset/send',json={'email':'unknown@example.org'})
            sender.assert_not_called()
            known=self.client.post('/api/auth/password/reset/send',json={'email':'person@example.org'})
        self.assertEqual(unknown.json(),known.json())

    def test_expired_attempt_limit_and_resend(self):
        _,payload=self.prepare()
        with server._get_db() as conn:
            conn.execute('UPDATE password_reset_codes SET expires=0');conn.commit()
        self.assertEqual(self.client.post('/api/auth/password/reset/confirm',json=payload).status_code,400)
        with server._get_db() as conn:
            conn.execute('UPDATE password_reset_codes SET expires=9999999999');conn.commit()
        wrong={**payload,'code':'999999' if payload['code']!='999999' else '000000'}
        for _ in range(5):
            self.assertEqual(self.client.post('/api/auth/password/reset/confirm',json=wrong).status_code,400)
        self.assertEqual(self.client.post('/api/auth/password/reset/confirm',json=payload).status_code,400)

    def test_missing_smtp_and_send_failure(self):
        with patch.dict(os.environ,{'SMTP_HOST':''}):
            self.assertEqual(self.client.post('/api/auth/password/reset/send',json={'email':'person@example.org'}).status_code,503)
        self.prepare();_budgets.clear()
        with patch.object(auth,'send_verification_email',side_effect=RuntimeError('private SMTP secret')):
            result=self.client.post('/api/auth/password/reset/send',json={'email':'person@example.org'})
        self.assertEqual(result.status_code,200)
        self.assertNotIn('secret',result.text)
        with server._get_db() as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM password_reset_codes').fetchone()[0],0)

    def test_rate_limit_and_password_change_invalidation(self):
        headers,payload=self.prepare()
        self.assertEqual(self.client.post('/api/auth/password/reset/send',json={'email':payload['email']}).status_code,429)
        self.assertEqual(self.client.post('/api/profile/password',headers=headers,json={'current_password':'test-password','new_password':'changed-password'}).status_code,200)
        self.assertEqual(self.client.post('/api/auth/password/reset/confirm',json=payload).status_code,400)
