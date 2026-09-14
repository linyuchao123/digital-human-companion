"""Bounded extractive session notes: no model calls, no persistent memory."""
import json

MAX_NOTES = 8
MAX_NOTE_CHARS = 160

def compact_context(messages, previous='', keep=38):
    try:
        notes=json.loads(previous) if previous else []
        if not isinstance(notes,list): notes=[]
        notes=[n[:MAX_NOTE_CHARS] for n in notes if isinstance(n,str)][-MAX_NOTES:]
    except (ValueError,TypeError):
        notes=[]
    history=list(messages)
    removed=history[:-keep] if len(history)>keep else []
    for message in removed:
        # Only quote user statements; never turn assistant claims into user facts.
        if message.role=='user' and message.content.strip():
            note=message.content.strip()[:MAX_NOTE_CHARS]
            if note not in notes: notes.append(note)
    notes=notes[-MAX_NOTES:]
    return history[-keep:], json.dumps(notes,ensure_ascii=False) if notes else ''
