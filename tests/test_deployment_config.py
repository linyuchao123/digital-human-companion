import unittest
from pathlib import Path


class DeploymentConfigTests(unittest.TestCase):
    def test_current_dependencies_and_private_persistence(self):
        root=Path(__file__).resolve().parents[1]
        docker=(root/'infra/docker/Dockerfile').read_text()
        compose=(root/'infra/docker/docker-compose.yml').read_text()
        ignored=(root/'.dockerignore').read_text()
        self.assertIn('COPY pyproject.toml',docker)
        self.assertNotIn('pip install -r requirements.txt',docker)
        self.assertIn('USER appuser',docker)
        self.assertIn('PUBLIC_DEPLOYMENT=true',compose)
        self.assertIn('127.0.0.1:8800:8800',compose)
        self.assertIn('app-data:/app/data',compose)
        self.assertIn('.venv*/',ignored)
        self.assertIn('data/users.db*',ignored)
