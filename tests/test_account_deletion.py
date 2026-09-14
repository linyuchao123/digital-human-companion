import sqlite3
from unittest.mock import patch
from tests.test_account_security import AccountSecurityTests
from apps.api import integrated_server as server
from apps.api.account_lifecycle import OWNED_TABLES


class AccountDeletionTests(AccountSecurityTests):
    payload={'current_password':'test-password','confirmation':'注销我的账号'}

    def delete(self,headers,payload=None):
        return self.client.post('/api/profile/account/delete',headers=headers,json=payload or self.payload)

    def test_requires_login_password_and_explicit_confirmation(self):
        self.assertEqual(self.delete({}).status_code,401)
        headers=self.register()
        self.assertEqual(self.delete(headers,{**self.payload,'confirmation':'注销'}).status_code,400)
        self.assertEqual(self.delete(headers,{**self.payload,'current_password':'wrong-password'}).status_code,403)
        self.assertEqual(self.client.get('/api/profile',headers=headers).status_code,200)
        with server._get_db() as conn:
            conn.execute('UPDATE users SET password_set=0');conn.commit()
        self.assertEqual(self.delete(headers).status_code,403)

    def seed(self,user_id):
        with server._get_db() as conn:
            conn.execute('INSERT INTO chat_sessions(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)',
                         (f's{user_id}',user_id,'private','2026','2026'))
            conn.execute('INSERT INTO chat_messages(session_id,role,content,ts,knowledge_sources) VALUES(?,?,?,?,?)',
                         (f's{user_id}','assistant','private','2026','[{"title":"private"}]'))
            for table in OWNED_TABLES:
                if table in ('auth_tokens','chat_sessions'):continue
                columns=list(conn.execute(f'PRAGMA table_info({table})'))
                values=[user_id if r['name']=='user_id' else f's{user_id}' if r['name']=='session_id' else (1 if r['type'] in ('INTEGER','REAL') else f'{user_id}-{r["name"]}') for r in columns]
                conn.execute(f'INSERT INTO {table} VALUES({",".join("?" for _ in values)})',values)
            conn.commit()

    def test_complete_owner_cleanup_and_other_account_preserved(self):
        headers=self.register();other=self.register('other-owner')
        owner=self.client.get('/api/profile',headers=headers).json()['id']
        other_id=self.client.get('/api/profile',headers=other).json()['id']
        self.seed(owner);self.seed(other_id)
        self.assertEqual(self.delete(headers).status_code,200)
        self.assertEqual(self.client.get('/api/profile',headers=headers).status_code,401)
        self.assertEqual(self.client.get('/api/profile',headers=other).status_code,200)
        with server._get_db() as conn:
            for table in OWNED_TABLES:
                self.assertEqual(conn.execute(f'SELECT COUNT(*) FROM {table} WHERE user_id=?',(owner,)).fetchone()[0],0,table)
                self.assertGreater(conn.execute(f'SELECT COUNT(*) FROM {table} WHERE user_id=?',(other_id,)).fetchone()[0],0,table)
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM chat_messages WHERE session_id=?',(f's{owner}',)).fetchone()[0],0)
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM chat_messages WHERE session_id=?',(f's{other_id}',)).fetchone()[0],1)
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute('INSERT INTO user_memories(id,user_id,content,created_at) VALUES(?,?,?,?)',('late',owner,'late','2026'))
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute('INSERT INTO chat_messages(session_id,role,content,ts) VALUES(?,?,?,?)',(f's{owner}','assistant','late','2026'))

    def test_transaction_rolls_back_on_cleanup_failure(self):
        headers=self.register()
        def fail(conn,user_id):
            conn.execute('DELETE FROM auth_tokens WHERE user_id=?',(user_id,))
            raise RuntimeError('simulated failure')
        with patch('apps.api.account_lifecycle.delete_owned_account',side_effect=fail):
            with self.assertRaises(RuntimeError):self.delete(headers)
        self.assertEqual(self.client.get('/api/profile',headers=headers).status_code,200)
