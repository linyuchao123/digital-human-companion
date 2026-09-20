import os
from unittest.mock import patch

from apps.api import external_auth as auth, integrated_server as server
from apps.api.security import _budgets
from tests.test_account_security import AccountSecurityTests


class EmailRegistrationTests(AccountSecurityTests):
    def setUp(self):
        super().setUp()
        self.env=patch.dict(os.environ,{
            'PUBLIC_DEPLOYMENT':'false','SMTP_HOST':'mail.example.org','SMTP_USERNAME':'test',
            'SMTP_PASSWORD':'test','SMTP_FROM':'test@example.org','EMAIL_VERIFICATION_SECRET':'x'*32,
        })
        self.env.start();_budgets.clear()

    def tearDown(self):
        self.env.stop();super().tearDown()

    def test_public_registration_requires_verified_email_and_supports_email_login(self):
        with patch.dict(os.environ,{'PUBLIC_DEPLOYMENT':'true'}):
            missing=self.client.post('/api/auth/register',json={'username':'email-user','password':'test-password'})
            self.assertEqual(missing.status_code,400)
            with patch.object(auth,'send_verification_email') as sender:
                sent=self.client.post('/api/auth/register/email/send',json={'email':'Person@example.org'})
                self.assertEqual(sent.status_code,200)
                code=sender.call_args.args[1]
                self.assertEqual(sender.call_args.args[2],'注册')
                self.assertNotIn(code,sent.text)
        with server._get_db() as conn:
            row=conn.execute('SELECT * FROM registration_email_codes').fetchone()
            self.assertNotEqual(row['digest'],code)
        with patch.dict(os.environ,{'PUBLIC_DEPLOYMENT':'true'}):
            registered=self.client.post('/api/auth/register',json={
                'username':'email-user','password':'test-password','email':'PERSON@example.org','email_code':code,
            })
        self.assertEqual(registered.status_code,200)
        with server._get_db() as conn:
            user=conn.execute("SELECT email,email_verified_at FROM users WHERE username='email-user'").fetchone()
            self.assertEqual(user['email'],'person@example.org');self.assertTrue(user['email_verified_at'])
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM registration_email_codes').fetchone()[0],0)
        login=self.client.post('/api/auth/login',json={'username':'person@example.org','password':'test-password'})
        self.assertEqual(login.status_code,200)

    def test_registration_code_is_one_time_and_wrong_code_is_rejected(self):
        with patch.object(auth,'send_verification_email') as sender:
            self.client.post('/api/auth/register/email/send',json={'email':'person@example.org'})
            code=sender.call_args.args[1]
        wrong='000000' if code!='000000' else '111111'
        payload={'username':'wrong-user','password':'test-password','email':'person@example.org','email_code':wrong}
        with patch.dict(os.environ,{'PUBLIC_DEPLOYMENT':'true'}):
            self.assertEqual(self.client.post('/api/auth/register',json=payload).status_code,400)
            payload.update(username='right-user',email_code=code)
            self.assertEqual(self.client.post('/api/auth/register',json=payload).status_code,200)
            payload.update(username='replay-user')
            self.assertEqual(self.client.post('/api/auth/register',json=payload).status_code,409)


if __name__ == '__main__':
    import unittest
    unittest.main()
