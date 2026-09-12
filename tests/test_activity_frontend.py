import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which("node"), "Node.js 未安装")
class ActivityFrontendTests(unittest.TestCase):
    def test_selection_and_completion_do_not_send_messages_or_execute_html(self):
        html = (Path(__file__).resolve().parents[1] / "integrated.html").read_text()
        function = re.search(r"function renderActivityCards\(.*?(?=function sendText)", html, re.S).group()
        script = """
        class Element{
          constructor(){this.children=[];this.dataset={};}
          append(...items){this.children.push(...items);}
          appendChild(item){this.children.push(item);}
        }
        const area=new Element();
        const document={createElement:()=>new Element(),getElementById:()=>area};
        """ + function + """
        renderActivityCards([null,{title:'<script>unsafe</script>',description:'记录心情',minutes:2}]);
        const card=area.children[0].children[0],button=card.children[2];
        button.onclick();const started=button.textContent;
        button.onclick();console.log(JSON.stringify({started,done:button.textContent,
          disabled:button.disabled,title:card.children[0].textContent}));
        """
        result = subprocess.run(["node"], input=script, text=True, capture_output=True, check=True)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["started"], "标记完成")
        self.assertEqual(payload["done"], "✓ 已完成")
        self.assertTrue(payload["disabled"])
        self.assertIn("<script>unsafe</script>", payload["title"])
        self.assertNotIn("innerHTML", function)
        self.assertNotIn("chatSend", function)
