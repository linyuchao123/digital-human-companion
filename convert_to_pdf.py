import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

async def convert_md_to_pdf(md_file, pdf_file):
    """将Markdown文件转换为PDF（简单HTML渲染）"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # 读取markdown内容
        content = md_file.read_text(encoding='utf-8')
        
        # 简单Markdown转HTML
        lines = content.split('\n')
        html_lines = []
        for line in lines:
            if line.startswith('### '):
                html_lines.append(f'<h3>{line[4:]}</h3>')
            elif line.startswith('## '):
                html_lines.append(f'<h2>{line[3:]}</h2>')
            elif line.startswith('# '):
                html_lines.append(f'<h1>{line[2:]}</h1>')
            elif line.startswith('- ') or line.startswith('* '):
                html_lines.append(f'<li>{line[2:]}</li>')
            elif line.startswith('```'):
                html_lines.append('<pre><code>')
            elif line.startswith('|'):
                # 简单表格处理
                cells = [c.strip() for c in line.split('|') if c.strip()]
                if cells:
                    html_lines.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
            elif line.strip() == '---':
                html_lines.append('<hr>')
            elif line.strip():
                # 加粗处理
                line = line.replace('**', '<strong>')
                html_lines.append(f'<p>{line}</p>')
            else:
                html_lines.append('<br>')
        
        html_content = '\n'.join(html_lines)
        
        # 完整HTML
        html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<style>
    @page {{ size: A4; margin: 2cm; }}
    body {{ 
        font-family: "楷体", "KaiTi", serif; 
        font-size: 11pt; 
        line-height: 1.6; 
        padding: 2cm;
        color: #333;
    }}
    h1 {{ font-size: 18pt; text-align: center; margin: 20pt 0; }}
    h2 {{ font-size: 14pt; border-bottom: 2px solid #1a5fa8; padding-bottom: 5pt; margin: 15pt 0 8pt 0; }}
    h3 {{ font-size: 12pt; margin: 10pt 0 5pt 0; }}
    p {{ margin: 5pt 0; text-indent: 2em; }}
    code {{ background: #f5f5f5; padding: 2px 5px; font-size: 10pt; }}
    pre {{ background: #f5f5f5; padding: 10pt; margin: 8pt 0; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; margin: 8pt 0; }}
    th, td {{ border: 1px solid #ddd; padding: 6pt; text-align: left; }}
    th {{ background: #f0f0f0; font-weight: bold; }}
    hr {{ border: none; border-top: 1px solid #ccc; margin: 15pt 0; }}
    li {{ margin-left: 20pt; }}
</style>
</head>
<body>
{html_content}
</body>
</html>'''
        
        await page.set_content(html, wait_until='networkidle')
        await page.pdf(
            path=str(pdf_file), 
            format='A4', 
            margin={'top': '2cm', 'right': '2cm', 'bottom': '2cm', 'left': '2cm'}
        )
        await browser.close()
    
    size = pdf_file.stat().st_size / 1024 / 1024
    print(f'✅ 生成成功: {pdf_file.name} ({size:.2f} MB)')

async def main():
    work_dir = Path(r'd:\AI数字人情感陪护项目\交付物_v2')
    
    # 转换Docker使用说明
    print('\n[1/2] 转换Docker部署包使用说明...')
    await convert_md_to_pdf(
        work_dir / '数智心伴_Docker镜像部署包_使用说明.md',
        work_dir / '数智心伴_Docker镜像部署包_使用说明.pdf'
    )
    
    # 转换ASR使用说明
    print('\n[2/2] 转换语音识别模型使用说明...')
    await convert_md_to_pdf(
        work_dir / '数智心伴_语音识别模型工程文件_使用说明.md',
        work_dir / '数智心伴_语音识别模型工程文件_使用说明.pdf'
    )
    
    print('\n' + '='*60)
    print('所有PDF生成完成！')
    print('='*60)

asyncio.run(main())
