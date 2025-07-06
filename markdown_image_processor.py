"""
Markdown图片处理器 - 只做本地图片分析和注释，无需 web_serves 依赖
"""
import os
import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from async_image_analysis import AsyncImageAnalysis

def update_markdown_with_analysis(markdown_content: str, image_analysis_results: Dict[str, Dict[str, Any]]) -> str:
    """
    更新Markdown内容，替换图片alt和添加描述
    """
    # 先处理嵌套图片链接: [![alt](img_url)](link_url)
    def replace_linked_image(match):
        img_url = match.group(2)
        normalized_url = img_url.replace("\\", "/")
        if normalized_url.startswith("./"):
            normalized_url = normalized_url[2:]
        result = image_analysis_results.get(img_url) or image_analysis_results.get(normalized_url)
        if result:
            title = result.get("title", "图片")
            description = result.get("description", "")
            new_img = f"![{title}]({img_url})"
            if description:
                # 修复重复描述的问题：只在图片下方添加一次描述
                return f"[{new_img}\n> {description}]({match.group(3)})"
            else:
                return f"[{new_img}]({match.group(3)})"
        return match.group(0)

    # 先处理嵌套图片链接 - 修复正则表达式，避免匹配到已经处理过的内容
    content = re.sub(r'\[!\[(.*?)\]\(([^)]+)\)\]\(([^)]+)\)', replace_linked_image, markdown_content)

    # 再处理普通图片 - 修改正则表达式，避免匹配到已经在链接内的图片
    def replace_image(match):
        original_path = match.group(1)
        normalized_path = original_path.replace("\\", "/")
        if normalized_path.startswith("./"):
            normalized_path = normalized_path[2:]
        result = image_analysis_results.get(original_path) or image_analysis_results.get(normalized_path)
        if result:
            title = result.get("title", "图片")
            description = result.get("description", "")
            new_image = f"![{title}]({original_path})"
            if description:
                new_image += f"\n> {description}"
            return new_image
        return match.group(0)
    # 使用负向前瞻，避免匹配到在链接内的图片
    content = re.sub(r'(?<!\[)!\[.*?\]\(([^)]+)\)', replace_image, content)
    return content


class MarkdownImageProcessor:
    """
    Markdown图片处理器 - 只分析本地图片并为其添加注释
    """
    def __init__(self, max_concurrent: int = 3, api_key: str = None, base_url: str = None, vision_model: str = None, provider: str = "zhipu", title_min_length: int = 10, description_min_length: int = 50):
        self.provider = provider
        self.vision_model = vision_model
        self.max_concurrent = max_concurrent
        self.api_key = api_key
        self.base_url = base_url
        self.image_analyzer = None
        self.title_min_length = title_min_length
        self.description_min_length = description_min_length
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def extract_images(self, markdown_content: str, markdown_file_dir: str) -> List[dict]:
        """
        提取所有需要处理的图片（远程图片 + 存在的本地图片）
        返回格式: [{"image_url": ...} 或 {"local_image_path": ...}]
        """
        image_pattern = r'!\[.*?\]\(([^)]+)\)'
        matches = re.findall(image_pattern, markdown_content)
        images = []
        for match in matches:
            url = match.strip()
            if url.startswith(('http://', 'https://')):
                images.append({"image_url": url})
            else:
                abs_path = os.path.abspath(os.path.join(markdown_file_dir, url))
                if os.path.exists(abs_path):
                    images.append({"local_image_path": abs_path, "rel_path": url})
                else:
                    self.logger.warning(f"本地图片文件不存在: {url}, 跳过")
        return images

    async def process_images(self, images: List[dict]) -> Dict[str, Dict[str, Any]]:
        if not images:
            self.logger.info("没有发现需要处理的图片")
            return {}
        self.logger.info(f"开始处理 {len(images)} 张图片...")
        # 只传递 image_url 或 local_image_path 字段
        image_sources = []
        key_map = []  # 用于结果回填
        for img in images:
            if "image_url" in img:
                image_sources.append({"image_url": img["image_url"]})
                key_map.append(img["image_url"])
            elif "local_image_path" in img:
                image_sources.append({"local_image_path": img["local_image_path"]})
                key_map.append(img.get("rel_path", img["local_image_path"]))
        analysis_results = await self.image_analyzer.analyze_multiple_images(image_sources)
        processed_results = {}
        for i, key in enumerate(key_map):
            analysis_result = analysis_results[i]
            processed_results[key] = {
                "title": analysis_result.get("title", ""),
                "description": analysis_result.get("description", ""),
                "error": analysis_result.get("error")
            }
            self.logger.info(f"处理完成: {key}")
        return processed_results

    async def process_markdown_content(self, markdown_content: str, markdown_file_dir: str) -> str:
        images = self.extract_images(markdown_content, markdown_file_dir)
        if not images:
            self.logger.info("没有发现需要处理的图片，返回原始内容")
            return markdown_content
        
        # 为每个图片添加上下文信息
        for img in images:
            img_url = img.get("image_url") or img.get("rel_path", "")
            if img_url:
                context = self.extract_context_around_image(markdown_content, img_url)
                img["context"] = context
        
        image_results = await self.process_images(images)
        updated_content = update_markdown_with_analysis(markdown_content, image_results)
        return updated_content

    async def process_markdown_file(self, input_file_path: str, output_file_path: str = None) -> str:
        try:
            with open(input_file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"输入文件不存在: {input_file_path}")
        markdown_dir = os.path.dirname(os.path.abspath(input_file_path))
        updated_content = await self.process_markdown_content(original_content, markdown_dir)
        if output_file_path is None:
            output_file_path = input_file_path
        output_dir = os.path.dirname(output_file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        self.logger.info(f"处理完成，输出文件: {output_file_path}")
        return updated_content

    async def close(self):
        await self.image_analyzer.close()

    async def __aenter__(self):
        self.image_analyzer = AsyncImageAnalysis(
            provider=self.provider,
            api_key=self.api_key,
            base_url=self.base_url,
            vision_model=self.vision_model,
            max_concurrent=self.max_concurrent
        )
        await self.image_analyzer.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.image_analyzer:
            await self.image_analyzer.__aexit__(exc_type, exc_val, exc_tb)

    def extract_context_around_image(self, markdown_content: str, image_url: str, context_length: int = 300) -> str:
        """
        提取图片周围的上下文文本
        """
        # 找到图片在markdown中的位置
        import re
        
        # 匹配图片的各种格式
        patterns = [
            rf'!\[.*?\]\({re.escape(image_url)}\)',
            rf'\[!\[.*?\]\({re.escape(image_url)}\)\]\([^)]+\)'
        ]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, markdown_content))
            if matches:
                match = matches[0]
                start_pos = match.start()
                end_pos = match.end()
                
                # 提取前后文本
                context_start = max(0, start_pos - context_length)
                context_end = min(len(markdown_content), end_pos + context_length)
                
                context_text = markdown_content[context_start:context_end]
                
                # 清理上下文文本，移除图片标记
                context_text = re.sub(r'!\[.*?\]\([^)]+\)', '', context_text)
                context_text = re.sub(r'\[!\[.*?\]\([^)]+\)\]\([^)]+\)', '', context_text)
                context_text = re.sub(r'\n+', '\n', context_text).strip()
                
                return context_text
        
        return ""


async def main():
    # 使用示例，请根据需要修改文件路径
    # 需要一个 .env 文件来配置 OPENAI_API_KEY 和 OPENAI_API_BASE
    # from dotenv import load_dotenv
    # load_dotenv()
    
    # processor = MarkdownImageProcessor()
    # await processor.process_markdown_file(
    #     "./assets/test.md",
    #     "./assets/test_updated.md"
    # )
    print("请在 main 函数中提供实际的文件路径以进行测试。")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
