import argparse
import asyncio
import os
from convert_md import main as convert_main


def main():
    parser = argparse.ArgumentParser(description="Markdown Image Auto-desc.")
    parser.add_argument("input", help="Input markdown file path.")
    parser.add_argument(
        "-o", "--output", help="Output markdown file path.", default=None
    )
    parser.add_argument(
        "--provider",
        choices=["zhipu", "openai"],
        default="zhipu",
        help="服务商，zhipu 或 openai，默认 zhipu",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="多模态模型名，如 glm-4v-flash 或 gpt-4-vision-preview",
    )
    args = parser.parse_args()

    input_path = args.input
    
    # 如果未提供输出路径，则在原文件名基础上添加 "_with_desc" 后缀
    if args.output:
        output_path = args.output
    else:
        # 分离文件名和扩展名
        base_name, ext = os.path.splitext(input_path)
        output_path = f"{base_name}_with_desc{ext}"

    asyncio.run(
        convert_main(
            input_path, output_path, provider=args.provider, model=args.model
        )
    )


if __name__ == "__main__":
    main()
