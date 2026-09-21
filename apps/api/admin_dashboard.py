"""Server-authorized operator dashboard and privacy-preserving usage accounting."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse


ROOT = Path(__file__).resolve().parents[2]


def _admin_names() -> set[str]:
    return {
        value.strip().lower()
        for value in os.getenv("ADMIN_USERNAMES", "").split(",")
        if value.strip()
    }


def configured_role(username: str) -> str:
    return "admin" if username.strip().lower() in _admin_names() else "user"


def init_admin_tables(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
    if "role" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS model_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            trace_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            request_kind TEXT NOT NULL,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cached_input_tokens INTEGER NOT NULL DEFAULT 0,
            estimated INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'success',
            estimated_cost_cny REAL NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_model_usage_user ON model_usage(user_id,created_at);
        CREATE INDEX IF NOT EXISTS idx_model_usage_provider ON model_usage(provider,created_at);
        CREATE TABLE IF NOT EXISTS user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            category TEXT NOT NULL,
            content TEXT NOT NULL,
            contact TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'new',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_user_feedback_created ON user_feedback(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_user_feedback_status ON user_feedback(status,created_at DESC);
        """
    )
    names = _admin_names()
    conn.execute("UPDATE users SET role='user' WHERE role='admin'")
    if names:
        placeholders = ",".join("?" for _ in names)
        conn.execute(
            f"UPDATE users SET role='admin' WHERE lower(username) IN ({placeholders})",
            tuple(sorted(names)),
        )


def user_role(conn: sqlite3.Connection, user_id: int) -> str:
    row = conn.execute("SELECT username,role FROM users WHERE id=?", (user_id,)).fetchone()
    if not row:
        return "user"
    return "admin" if row[1] == "admin" or row[0].lower() in _admin_names() else "user"


def _price(provider: str, kind: str) -> float:
    prefix = provider.upper().replace("-", "_")
    value = os.getenv(f"{prefix}_{kind}_PRICE_CNY_PER_MILLION", "0").strip()
    try:
        return max(0.0, float(value or 0))
    except ValueError:
        return 0.0


def pricing_status() -> dict[str, dict[str, float | bool]]:
    result: dict[str, dict[str, float | bool]] = {}
    for provider in ("deepseek", "qwen"):
        input_price = _price(provider, "INPUT")
        cached_input_price = _price(provider, "CACHED_INPUT")
        output_price = _price(provider, "OUTPUT")
        result[provider] = {
            "input_cny_per_million": input_price,
            "cached_input_cny_per_million": cached_input_price,
            "output_cny_per_million": output_price,
            "configured": bool(input_price or output_price),
        }
    return result


def save_model_usage(
    get_db: Callable[[], sqlite3.Connection],
    rows: list[dict[str, Any]],
    *,
    user_id: int,
    session_id: str,
    trace_id: str,
) -> None:
    if not rows:
        return
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    conn = get_db()
    try:
        for row in rows:
            provider = str(row.get("provider", "unknown"))[:32].lower()
            input_tokens = max(0, int(row.get("input_tokens", 0) or 0))
            output_tokens = max(0, int(row.get("output_tokens", 0) or 0))
            cached_input_tokens = min(
                input_tokens, max(0, int(row.get("cached_input_tokens", 0) or 0))
            )
            cost = (
                (input_tokens - cached_input_tokens) * _price(provider, "INPUT")
                + cached_input_tokens * _price(provider, "CACHED_INPUT")
                + output_tokens * _price(provider, "OUTPUT")
            ) / 1_000_000
            conn.execute(
                """INSERT INTO model_usage(
                    user_id,session_id,trace_id,provider,model,request_kind,
                    input_tokens,output_tokens,cached_input_tokens,estimated,status,
                    estimated_cost_cny,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    user_id,
                    session_id,
                    trace_id,
                    provider,
                    str(row.get("model", ""))[:128],
                    str(row.get("request_kind", "chat"))[:32],
                    input_tokens,
                    output_tokens,
                    cached_input_tokens,
                    1 if row.get("estimated") else 0,
                    "success" if row.get("status") == "success" else "failed",
                    round(cost, 8),
                    now,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def install_admin_routes(
    app: FastAPI,
    get_db: Callable[[], sqlite3.Connection],
    verify_token: Callable[[str], int | None],
) -> None:
    def request_user_id(request: Request) -> int | None:
        token = request.headers.get("X-Auth-Token", "") or request.cookies.get("auth_token", "")
        return verify_token(token)

    def admin_id(request: Request) -> int | None:
        user_id = request_user_id(request)
        if user_id is None:
            return None
        with get_db() as conn:
            return user_id if user_role(conn, user_id) == "admin" else None

    def forbidden(request: Request) -> JSONResponse | None:
        if request_user_id(request) is None:
            return JSONResponse({"error": "请先登录"}, status_code=401)
        if admin_id(request) is None:
            return JSONResponse({"error": "仅管理员可访问"}, status_code=403)
        return None

    @app.post("/api/feedback")
    async def submit_feedback(request: Request):
        user_id = request_user_id(request)
        if user_id is None:
            return JSONResponse({"error": "请先登录后提交反馈"}, status_code=401)
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "反馈格式无效"}, status_code=400)
        category = str(payload.get("category", "suggestion")).strip().lower()
        content = str(payload.get("content", "")).strip()
        contact = str(payload.get("contact", "")).strip()
        if category not in {"bug", "suggestion", "experience", "other"}:
            return JSONResponse({"error": "请选择有效的反馈类型"}, status_code=400)
        if len(content) < 5:
            return JSONResponse({"error": "请至少填写 5 个字，让我们更好地理解你的想法"}, status_code=400)
        if len(content) > 2000 or len(contact) > 160:
            return JSONResponse({"error": "反馈内容或联系方式过长"}, status_code=400)
        created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        with get_db() as conn:
            cursor = conn.execute(
                """INSERT INTO user_feedback(user_id,category,content,contact,status,created_at)
                   VALUES(?,?,?,?,?,?)""",
                (user_id, category, content, contact, "new", created_at),
            )
            conn.commit()
            feedback_id = cursor.lastrowid
        return {"ok": True, "feedback_id": feedback_id, "message": "感谢你的反馈，我们会认真阅读。"}

    @app.get("/admin", include_in_schema=False)
    async def admin_page(request: Request):
        denied = forbidden(request)
        if denied:
            return denied
        return FileResponse(ROOT / "admin.html", media_type="text/html")

    @app.get("/api/admin/summary")
    async def admin_summary(request: Request):
        denied = forbidden(request)
        if denied:
            return denied
        with get_db() as conn:
            totals = conn.execute(
                """SELECT
                    (SELECT COUNT(*) FROM users) AS users,
                    (SELECT COUNT(*) FROM users WHERE datetime(created_at)>=datetime('now','-7 day')) AS new_users_7d,
                    (SELECT COUNT(DISTINCT s.user_id) FROM chat_messages m JOIN chat_sessions s ON s.id=m.session_id
                        WHERE datetime(m.ts)>=datetime('now','-7 day')) AS active_users_7d,
                    (SELECT COUNT(*) FROM chat_sessions) AS sessions,
                    (SELECT COUNT(*) FROM chat_messages WHERE role='user') AS dialogue_rounds,
                    (SELECT COUNT(*) FROM model_usage) AS model_calls,
                    (SELECT COUNT(*) FROM model_usage WHERE status='failed') AS failed_calls,
                    (SELECT COALESCE(SUM(input_tokens),0) FROM model_usage) AS input_tokens,
                    (SELECT COALESCE(SUM(output_tokens),0) FROM model_usage) AS output_tokens,
                    (SELECT COALESCE(SUM(estimated_cost_cny),0) FROM model_usage) AS estimated_cost_cny
                """
            ).fetchone()
            providers = [dict(row) for row in conn.execute(
                """SELECT provider,COUNT(*) calls,
                    COALESCE(SUM(input_tokens),0) input_tokens,
                    COALESCE(SUM(output_tokens),0) output_tokens,
                    ROUND(COALESCE(SUM(estimated_cost_cny),0),6) estimated_cost_cny
                    FROM model_usage GROUP BY provider ORDER BY calls DESC"""
            )]
            daily = [dict(row) for row in conn.execute(
                """WITH RECURSIVE days(day) AS (
                    SELECT date('now','-13 day') UNION ALL
                    SELECT date(day,'+1 day') FROM days WHERE day<date('now')
                ) SELECT day,
                    (SELECT COUNT(*) FROM users WHERE date(created_at)=day) registrations,
                    (SELECT COUNT(*) FROM chat_messages WHERE role='user' AND date(ts)=day) dialogue_rounds,
                    (SELECT COUNT(*) FROM model_usage WHERE date(created_at)=day) model_calls
                FROM days"""
            )]
        return {
            "totals": dict(totals),
            "providers": providers,
            "daily": daily,
            "pricing": pricing_status(),
            "privacy": "仅展示运营统计，不返回聊天原文、密码、验证码或 API Key。",
        }

    @app.get("/api/admin/users")
    async def admin_users(
        request: Request,
        search: str = Query(default="", max_length=80),
        limit: int = Query(default=100, ge=1, le=200),
    ):
        denied = forbidden(request)
        if denied:
            return denied
        pattern = f"%{search.strip()}%"
        with get_db() as conn:
            rows = conn.execute(
                """SELECT u.id,u.username,u.email,u.role,u.created_at,
                    MAX(COALESCE(m.ts,u.created_at)) last_active_at,
                    COUNT(DISTINCT s.id) sessions,
                    COUNT(DISTINCT CASE WHEN m.role='user' THEN m.id END) dialogue_rounds
                FROM users u
                LEFT JOIN chat_sessions s ON s.user_id=u.id
                LEFT JOIN chat_messages m ON m.session_id=s.id
                WHERE (?='' OR u.username LIKE ? OR u.email LIKE ?)
                GROUP BY u.id ORDER BY last_active_at DESC LIMIT ?""",
                (search.strip(), pattern, pattern, limit),
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                usage = conn.execute(
                    """SELECT COUNT(*) model_calls,COALESCE(SUM(input_tokens),0) input_tokens,
                        COALESCE(SUM(output_tokens),0) output_tokens,
                        ROUND(COALESCE(SUM(estimated_cost_cny),0),6) estimated_cost_cny
                        FROM model_usage WHERE user_id=?""",
                    (item["id"],),
                ).fetchone()
                item.update(dict(usage))
                result.append(item)
        return {"users": result, "count": len(result)}

    @app.get("/api/admin/feedback")
    async def admin_feedback(
        request: Request,
        status: str = Query(default="", max_length=20),
        limit: int = Query(default=100, ge=1, le=200),
    ):
        denied = forbidden(request)
        if denied:
            return denied
        status = status.strip().lower()
        if status and status not in {"new", "reviewed", "resolved"}:
            return JSONResponse({"error": "反馈状态无效"}, status_code=400)
        with get_db() as conn:
            rows = conn.execute(
                """SELECT f.id,f.category,f.content,f.contact,f.status,f.created_at,
                          u.username,u.email
                   FROM user_feedback f JOIN users u ON u.id=f.user_id
                   WHERE (?='' OR f.status=?)
                   ORDER BY f.id DESC LIMIT ?""",
                (status, status, limit),
            ).fetchall()
        return {"feedback": [dict(row) for row in rows], "count": len(rows)}

    @app.patch("/api/admin/feedback/{feedback_id}")
    async def update_feedback_status(feedback_id: int, request: Request):
        denied = forbidden(request)
        if denied:
            return denied
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "反馈状态格式无效"}, status_code=400)
        status = str(payload.get("status", "")).strip().lower()
        if status not in {"new", "reviewed", "resolved"}:
            return JSONResponse({"error": "反馈状态无效"}, status_code=400)
        with get_db() as conn:
            cursor = conn.execute("UPDATE user_feedback SET status=? WHERE id=?", (status, feedback_id))
            conn.commit()
        if not cursor.rowcount:
            return JSONResponse({"error": "反馈不存在"}, status_code=404)
        return {"ok": True, "status": status}
