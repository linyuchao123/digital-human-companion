"""Current-account deletion only; shared knowledge and model assets are excluded."""
OWNED_TABLES=('auth_tokens','user_memory_settings','user_memories','agent_runs','activity_tasks',
              'social_accounts','oauth_states','email_codes','password_reset_codes','session_notes',
              'model_usage','chat_sessions')


def install_owner_guards(conn):
    # SQLite existing schemas cannot gain foreign keys through ALTER TABLE.
    # Guard new writes only; no sweeping cleanup of legacy/user data at startup.
    for table in OWNED_TABLES:
        conn.execute(f'''CREATE TRIGGER IF NOT EXISTS guard_{table}_account_insert
            BEFORE INSERT ON {table}
            WHEN NEW.user_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM users WHERE id=NEW.user_id)
            BEGIN SELECT RAISE(ABORT,'account_unavailable'); END''')
    conn.execute('''CREATE TRIGGER IF NOT EXISTS guard_message_session_insert
        BEFORE INSERT ON chat_messages
        WHEN NOT EXISTS(SELECT 1 FROM chat_sessions WHERE id=NEW.session_id)
        BEGIN SELECT RAISE(ABORT,'session_unavailable'); END''')


def delete_owned_account(conn,user_id):
    """Caller verifies password and owns one IMMEDIATE transaction."""
    conn.execute('DELETE FROM chat_messages WHERE session_id IN (SELECT id FROM chat_sessions WHERE user_id=?)',(user_id,))
    for table in OWNED_TABLES:
        conn.execute(f'DELETE FROM {table} WHERE user_id=?',(user_id,))
    conn.execute('DELETE FROM users WHERE id=?',(user_id,))
