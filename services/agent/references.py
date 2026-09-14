"""Public knowledge references, not proof that every generated claim is verified."""
from .web_search import safe_url
import re

def reference_snapshot(value):
    """Validate bounded public snapshots on both database write and read."""
    if not isinstance(value, list):
        return []
    result=[]
    for item in value[:3]:
        if not isinstance(item, dict) or not isinstance(item.get('source'), str) or not isinstance(item.get('excerpt'), str):
            continue
        url=item.get('url')
        result.append({'id':item.get('id')[:120] if isinstance(item.get('id'),str) else None,
                       'source':item['source'][:200], 'excerpt':item['excerpt'][:400],
                       'url':url[:500] if isinstance(url,str) and url.startswith('https://') and safe_url(url) else None})
    return result

def normalize_knowledge_citations(text, snippets):
    """Remove nonexistent numeric citation labels without fabricating evidence."""
    count=min(len(snippets),3)
    return re.sub(r'\[(\d{1,3})\]', lambda match:match[0] if 1<=int(match[1])<=count else '',text)

def knowledge_references(snippets):
    result=[]
    for item in snippets[:3]:
        url=item.source_url
        result.append({'id':item.document_id, 'source':item.source,
                       'excerpt':item.content[:400],
                       'url':url if url and url.startswith('https://') and safe_url(url) else None})
    return result
