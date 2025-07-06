import argparse
import asyncio
import os

def main():
    parser = argparse.ArgumentParser(description="Markdown Image Auto-desc.")
    parser.add_argument("-i", "--input", help="Input markdown file path.")
    parser.add_argument(
        "-o", "--output", help="Output markdown file path.", default=None
    )
    parser.add_argument(
        "-p", "--provider",
        choices=["zhipu", "openai"],
        default="zhipu",
        help="服务商，zhipu 或 openai，默认 zhipu",
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help="多模态模型名，如 glm-4v-flash 或 gpt-4-vision-preview",
    )
    parser.add_argument(
        "-t", "--title-min-length",
        type=int,
        default=10,
        help="标题最小字数，默认10",
    )
    parser.add_argument(
        "-d", "--desc-min-length",
        type=int,
        default=50,
        help="描述最小字数，默认50",
    )
    args = parser.parse_args()

    input_path = args.input
    # 如果未提供输出路径，则在原文件名基础上添加 "_with_desc" 后缀
    if args.output:
        output_path = args.output
    else:
        base_name, ext = os.path.splitext(input_path)
        output_path = f"{base_name}_with_desc{ext}"

    from dotenv import load_dotenv
    from markdown_image_processor import MarkdownImageProcessor
    load_dotenv()
    async def run():
        async with MarkdownImageProcessor(
            provider=args.provider,
            vision_model=args.model,
            max_concurrent=3,
            title_min_length=args.title_min_length,
            description_min_length=args.desc_min_length
        ) as processor:
            await processor.process_markdown_file(input_path, output_path)
    asyncio.run(run())


if __name__ == "__main__":
    main()
