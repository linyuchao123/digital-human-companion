import tempfile
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from apps.api import integrated_server as server


class ActivityHistoryTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.original=server.DB_PATH
        server.DB_PATH=Path(self.folder.name)/"test.db"
        server._init_db()
        self.client=TestClient(server.app)
        self.headers=[]
        for name in ("activity-first", "activity-second"):
            data=self.client.post('/api/auth/register',json={"username":name,"password":"test-password"}).json()
            self.headers.append({'X-Auth-Token':data['token']})

    def tearDown(self):
        self.client.close()
        server.DB_PATH=self.original
        self.folder.cleanup()

    def test_idempotency_history_and_user_isolation(self):
        payload={'template_id':'tidy','client_id':'request-one'}
        task=self.client.post('/api/activities',json=payload,headers=self.headers[0]).json()
        duplicate=self.client.post('/api/activities',json=payload,headers=self.headers[0]).json()
        self.assertEqual(task['task_id'],duplicate['task_id'])
        url='/api/activities/'+task['task_id']+'/complete'
        self.assertEqual(self.client.post(url,headers=self.headers[1]).status_code,404)
        first=self.client.post(url,headers=self.headers[0]).json()
        second=self.client.post(url,headers=self.headers[0]).json()
        self.assertEqual(first['completed_at'],second['completed_at'])
        history=self.client.get('/api/activities',headers=self.headers[0]).json()['activities']
        self.assertEqual(len(history),1)
        self.assertEqual(history[0]['status'],'completed')
        self.assertEqual(self.client.get('/api/activities',headers=self.headers[1]).json()['activities'],[])
        self.assertEqual(self.client.post('/api/activities',json={**payload,'template_id':'music'},headers=self.headers[0]).status_code,409)

    def test_requires_auth_and_valid_template(self):
        self.assertEqual(self.client.get('/api/activities').status_code,401)
        self.assertEqual(self.client.post('/api/activities',json={'template_id':'tidy','client_id':'x'}).status_code,401)
        self.assertEqual(self.client.post('/api/activities/not-mine/complete').status_code,401)
        self.assertEqual(self.client.post('/api/activities',json={'template_id':'injected','client_id':'x'},headers=self.headers[0]).status_code,422)

    def test_reinitializing_database_keeps_tasks(self):
        self.client.post('/api/activities',json={'template_id':'journal','client_id':'stable'},headers=self.headers[0])
        server._init_db()
        self.assertEqual(len(self.client.get('/api/activities',headers=self.headers[0]).json()['activities']),1)

    def test_summary_and_removal_are_scoped_and_idempotent(self):
        headers=self.headers[0]
        self.assertEqual(self.client.get('/api/activities/summary').status_code,401)
        self.assertEqual(self.client.get('/api/activities/summary',headers=headers).json(),{'total':0,'completed':0,'started':0})
        payload={'template_id':'tidy','client_id':'review'}
        task=self.client.post('/api/activities',json=payload,headers=headers).json()
        path='/api/activities/'+task['task_id']
        self.assertEqual(self.client.delete(path,headers=self.headers[1]).status_code,404)
        self.client.post(path+'/complete',headers=headers)
        self.assertEqual(self.client.get('/api/activities/summary',headers=headers).json(),{'total':1,'completed':1,'started':0})
        self.assertEqual(self.client.delete(path,headers=headers).status_code,200)
        self.assertEqual(self.client.delete(path,headers=headers).status_code,200)
        self.assertEqual(self.client.get('/api/activities/summary',headers=headers).json()['total'],0)
        self.assertEqual(self.client.get('/api/activities',headers=headers).json()['activities'],[])
        self.assertEqual(self.client.post(path+'/complete',headers=headers).status_code,404)
        self.assertEqual(self.client.post('/api/activities',json=payload,headers=headers).status_code,410)

    def test_schema_migration_preserves_existing_tasks(self):
        conn=server._get_db()
        conn.execute('ALTER TABLE activity_tasks DROP COLUMN removed_at')
        conn.commit();conn.close()
        server._init_db()
        self.assertEqual(self.client.get('/api/activities',headers=self.headers[0]).status_code,200)
