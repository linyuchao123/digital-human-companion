"""Consent-gated, bounded extractive notes. Never infer facts or call a model."""
import json
from .memory import extract_memory_candidate
from .safety import SafetyTriage

MAX_NOTES=8
KEEP_MESSAGES=38


def init_session_notes(conn):
    columns={row[1] for row in conn.execute('PRAGMA table_info(user_memory_settings)')}
    if 'revision' not in columns:
        conn.execute('ALTER TABLE user_memory_settings ADD COLUMN revision INTEGER NOT NULL DEFAULT 0')
    message_columns={row[1] for row in conn.execute('PRAGMA table_info(chat_messages)')}
    if 'memory_revision' not in message_columns:
        conn.execute('ALTER TABLE chat_messages ADD COLUMN memory_revision INTEGER')
    conn.execute('''CREATE TABLE IF NOT EXISTS session_notes(
        session_id TEXT PRIMARY KEY,user_id INTEGER NOT NULL,
        notes TEXT NOT NULL DEFAULT '[]',through_id INTEGER NOT NULL DEFAULT 0)''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_session_notes_user ON session_notes(user_id)')
    conn.execute('''CREATE TRIGGER IF NOT EXISTS guard_notes_session_insert
        BEFORE INSERT ON session_notes WHEN NOT EXISTS(
            SELECT 1 FROM chat_sessions WHERE id=NEW.session_id AND user_id=NEW.user_id)
        BEGIN SELECT RAISE(ABORT,'session_unavailable'); END''')


def reset_session_notes(conn,user_id):
    """Caller owns transaction. Advance watermark so forgotten history stays forgotten."""
    # Standalone memory-store tests/deployments may not include chat tables.
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='session_notes'").fetchone():return
    conn.execute('UPDATE user_memory_settings SET revision=revision+1 WHERE user_id=?',(user_id,))
    conn.execute('''INSERT INTO session_notes(session_id,user_id,notes,through_id)
        SELECT s.id,s.user_id,'[]',COALESCE((SELECT MAX(m.id) FROM chat_messages m WHERE m.session_id=s.id),0)
        FROM chat_sessions s WHERE s.user_id=?
        ON CONFLICT(session_id) DO UPDATE SET notes='[]',through_id=excluded.through_id''',(user_id,))


def load_session_notes(conn,session_id,user_id):
    settings=conn.execute('SELECT enabled,revision FROM user_memory_settings WHERE user_id=?',(user_id,)).fetchone()
    if not settings or not settings['enabled']:return '',None
    row=conn.execute('''SELECT n.notes FROM session_notes n JOIN chat_sessions s ON s.id=n.session_id
        WHERE n.session_id=? AND n.user_id=? AND s.user_id=?''',(session_id,user_id,user_id)).fetchone()
    return (row['notes'] if row else ''),settings['revision']


def update_session_notes(conn,session_id,user_id,revision):
    """Incremental scan capped at 200 rows; stale consent revisions cannot write."""
    if revision is None:return
    conn.execute('BEGIN IMMEDIATE')
    settings=conn.execute('SELECT enabled,revision FROM user_memory_settings WHERE user_id=?',(user_id,)).fetchone()
    owner=conn.execute('SELECT user_id FROM chat_sessions WHERE id=?',(session_id,)).fetchone()
    if not settings or not settings['enabled'] or settings['revision']!=revision or not owner or owner['user_id']!=user_id:return
    row=conn.execute('SELECT notes,through_id FROM session_notes WHERE session_id=?',(session_id,)).fetchone()
    try:
        notes=json.loads(row['notes']) if row else []
        if not isinstance(notes,list):notes=[]
        notes=[n[:200] for n in notes if isinstance(n,str)][-MAX_NOTES:]
    except (ValueError,TypeError):notes=[]
    through=row['through_id'] if row else 0
    cutoff=conn.execute('SELECT id FROM chat_messages WHERE session_id=? ORDER BY id DESC LIMIT 1 OFFSET ?',
                        (session_id,KEEP_MESSAGES)).fetchone()
    if not cutoff:return
    rows=conn.execute('''SELECT id,role,content,memory_revision FROM chat_messages
        WHERE session_id=? AND id>? AND id<=? ORDER BY id LIMIT 200''',(session_id,through,cutoff['id'])).fetchall()
    for message in rows:
        candidate=extract_memory_candidate(message['content']) if (
            message['role']=='user' and message['memory_revision']==revision
            and SafetyTriage().evaluate(message['content']).allow_memory_write) else None
        if candidate:
            label={'preference':'偏好','goal':'目标','profile':'称呼与资料','context':'日常背景'}[candidate.category]
            note=f'{label}：{candidate.content[:160]}'
            if note not in notes:notes.append(note)
        through=message['id']
    if rows:
        conn.execute('''INSERT INTO session_notes(session_id,user_id,notes,through_id) VALUES(?,?,?,?)
            ON CONFLICT(session_id) DO UPDATE SET notes=excluded.notes,through_id=excluded.through_id''',
            (session_id,user_id,json.dumps(notes[-MAX_NOTES:],ensure_ascii=False),through))
        conn.commit()
