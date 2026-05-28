"""
使用Playwright将Markdown文件转换为高质量PDF
先转换为完整HTML，保留所有格式
"""
import asyncio
from pathlib import Path
import re
from playwright.async_api import async_playwright

def markdown_to_html(md_content):
    """将Markdown转换为完整HTML，保留所有格式"""
    lines = md_content.split('\n')
    html = []
    in_code_block = False
    in_table = False
    in_list = False
    
    html.append('''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<style>
@page {
    size: A4;
    margin: 2cm 2cm 2cm 2cm;
}
body {
    font-family: "楷体", "KaiTi", "STKaiti", serif;
    font-size: 11pt;
    line-height: 1.8;
    color: #000;
    padding: 2cm;
}
h1 {
    font-size: 18pt;
    font-weight: bold;
    text-align: center;
    margin: 30pt 0 15pt 0;
    page-break-after: avoid;
}
h2 {
    font-size: 14pt;
    font-weight: bold;
    color: #1a5fa8;
    border-bottom: 2px solid #1a5fa8;
    padding-bottom: 5pt;
    margin: 20pt 0 10pt 0;
    page-break-after: avoid;
}
h3 {
    font-size: 12pt;
    font-weight: bold;
    margin: 12pt 0 6pt 0;
    page-break-after: avoid;
}
p {
    margin: 6pt 0;
    text-indent: 2em;
    text-align: justify;
}
ul, ol {
    margin: 6pt 0;
    padding-left: 2em;
}
li {
    margin: 3pt 0;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin: 10pt 0;
    page-break-inside: avoid;
}
th, td {
    border: 1px solid #333;
    padding: 5pt 8pt;
    text-align: left;
    font-size: 10pt;
}
th {
    background-color: #f0f0f0;
    font-weight: bold;
}
pre {
    background: #f5f5f5;
    border-left: 3px solid #1a5fa8;
    padding: 10pt;
    margin: 8pt 0;
    overflow-x: auto;
    font-size: 10pt;
    line-height: 1.5;
}
code {
    background: #f5f5f5;
    padding: 1pt 4pt;
    font-size: 10pt;
    border-radius: 2pt;
}
hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 15pt 0;
}
.blockquote {
    border-left: 4px solid #1a5fa8;
    padding-left: 10pt;
    margin: 8pt 0;
    color: #555;
}
</style>
</head>
<body>
''')
    
    for line in lines:
        # 代码块
        if line.startswith('```'):
            if in_code_block:
                html.append('</code></pre>')
                in_code_block = False
            else:
                html.append('<pre><code>')
                in_code_block = True
            continue
        
        if in_code_block:
            html.append(line.replace('<', '&lt;').replace('>', '&gt;'))
            continue
        
        # 标题
        if line.startswith('### '):
            html.append(f'<h3>{line[4:]}</h3>')
        elif line.startswith('## '):
            html.append(f'<h2>{line[3:]}</h2>')
        elif line.startswith('# '):
            html.append(f'<h1>{line[2:]}</h1>')
        
        # 表格
        elif line.startswith('|'):
            if not in_table:
                html.append('<table>')
                in_table = True
            cells = [c.strip() for c in line.split('|') if c.strip()]
            if cells and all(c.replace('-', '').strip() == '' for c in cells):
                continue  # 跳过分隔行
            tag = 'th' if len(html) > 0 and '<table>' in html[-2] else 'td'
            html.append('<tr>' + ''.join(f'<{tag}>{c}</{tag}>' for c in cells) + '</tr>')
        
        # 列表
        elif line.startswith('- ') or line.startswith('* ') or re.match(r'^\d+\.\s', line):
            if not in_list:
                html.append('<ul>')
                in_list = True
            content = re.sub(r'^[-*]|\d+\.\s', '', line).strip()
            html.append(f'<li>{content}</li>')
        
        # 分隔线
        elif line.strip() == '---':
            if in_list:
                html.append('</ul>')
                in_list = False
            if in_table:
                html.append('</table>')
                in_table = False
            html.append('<hr>')
        
        # 段落
        elif line.strip():
            if in_list:
                html.append('</ul>')
                in_list = False
            if in_table:
                html.append('</table>')
                in_table = False
            # 处理加粗
            line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
            html.append(f'<p>{line}</p>')
        
        else:
            if in_list:
                html.append('</ul>')
                in_list = False
            if in_table:
                html.append('</table>')
                in_table = False
    
    # 关闭未闭合的标签
    if in_list:
        html.append('</ul>')
    if in_table:
        html.append('</table>')
    if in_code_block:
        html.append('</code></pre>')
    
    html.append('</body></html>')
    return '\n'.join(html)

async def convert_md_to_pdf(md_file, pdf_file):
    """转换Markdown到PDF"""
    print(f'\n处理: {md_file.name}')
    
    # 读取Markdown
    md_content = md_file.read_text(encoding='utf-8')
    
    # 转换为HTML
    html_content = markdown_to_html(md_content)
    
    # 使用Playwright生成PDF
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html_content, wait_until='networkidle')
        await page.pdf(
            path=str(pdf_file),
            format='A4',
            margin={'top': '2cm', 'right': '2cm', 'bottom': '2cm', 'left': '2cm'},
            print_background=True
        )
        await browser.close()
    
    size = pdf_file.stat().st_size / 1024 / 1024
    print(f'✅ 生成成功: {pdf_file.name} ({size:.2f} MB)')

async def main():
    work_dir = Path(r'd:\AI数字人情感陪护项目\交付物_v2')
    
    print('=' * 60)
    print('PDF文档重新生成')
    print('=' * 60)
    
    # 转换Docker使用说明
    await convert_md_to_pdf(
        work_dir / '数智心伴_Docker镜像部署包_使用说明.md',
        work_dir / '数智心伴_Docker镜像部署包_使用说明_正式版.pdf'
    )
    
    # 转换ASR使用说明
    await convert_md_to_pdf(
        work_dir / '数智心伴_语音识别模型工程文件_使用说明.md',
        work_dir / '数智心伴_语音识别模型工程文件_使用说明_正式版.pdf'
    )
    
    print('\n' + '=' * 60)
    print('所有PDF生成完成！')
    print('=' * 60)

if __name__ == '__main__':
    asyncio.run(main())
