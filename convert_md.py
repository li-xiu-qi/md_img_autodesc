import asyncio
import os
from dotenv import load_dotenv
from markdown_image_processor import MarkdownImageProcessor

async def main(input_path, output_path, provider=None, model=None):
    # 加载 .env 文件中的环境变量，这样才能读取到 OPENAI_API_KEY/ZHIPU_API_KEY 等
    load_dotenv()
    
    async with MarkdownImageProcessor(provider=provider, vision_model=model) as processor:
        await processor.process_markdown_file(input_path, output_path)

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Markdown图片批量自动注释")
    parser.add_argument("input", help="输入的 Markdown 文件路径")
    parser.add_argument("output", help="输出的 Markdown 文件路径")
    parser.add_argument("--provider", choices=["zhipu", "openai"], default="zhipu", help="服务商，zhipu 或 openai，默认 zhipu")
    parser.add_argument("--model", default=None, help="多模态模型名，如 glm-4v 或 gpt-4-vision-preview")
    args = parser.parse_args()
    asyncio.run(main(args.input, args.output, provider=args.provider, model=args.model))
