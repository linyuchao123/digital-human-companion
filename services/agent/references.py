"""Public knowledge references, not proof that every generated claim is verified."""
from .web_search import safe_url

def knowledge_references(snippets):
    result=[]
    for item in snippets[:3]:
        url=item.source_url
        result.append({'id':item.document_id, 'source':item.source,
                       'excerpt':item.content[:400],
                       'url':url if url and url.startswith('https://') and safe_url(url) else None})
    return result
