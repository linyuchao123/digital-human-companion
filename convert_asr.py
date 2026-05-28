import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

async def convert_simple(md_file, pdf_file):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        content = md_file.read_text(encoding='utf-8')
        html = '<html><body style="font-family:楷体;font-size:11pt;line-height:1.6;padding:2cm">' + content.replace('\n', '<br>') + '</body></html>'
        await page.set_content(html)
        await page.pdf(path=str(pdf_file), format='A4')
        await browser.close()
    print(f'OK: {pdf_file.name} ({pdf_file.stat().st_size/1024/1024:.2f}MB)')

asyncio.run(convert_simple(
    Path(r'd:\AI数字人情感陪护项目\交付物_v2\数智心伴_语音识别模型工程文件_使用说明.md'),
    Path(r'd:\AI数字人情感陪护项目\交付物_v2\数智心伴_语音识别模型工程文件_使用说明.pdf')
))
