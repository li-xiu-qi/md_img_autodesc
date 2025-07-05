#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图片分析工具类 - 仅包含本地图片base64和json内容提取，无外部依赖
"""
import json
from typing import Dict, Any, Optional
import aiofiles
import base64


def extract_json_content(text: str) -> Dict[str, Any]:
    """
    从文本中提取JSON内容。
    参数:
        text (str): 可能包含JSON的文本
    返回:
        Dict[str, Any]: 解析后的JSON字典，如果解析失败则返回包含错误信息的字典
    """
    if not text:
        return {"error": "Empty response", "title": "", "description": ""}

    # 尝试寻找JSON的开始和结束位置
    json_start = text.find("{")
    json_end = text.rfind("}")
    if (json_start != -1 and json_end != -1 and json_end > json_start):
        try:
            json_text = text[json_start: json_end + 1]
            result = json.loads(json_text)
            if "title" not in result:
                result["title"] = ""
            if "description" not in result:
                result["description"] = ""
            return result
        except json.JSONDecodeError as e:
            return {"error": f"JSON解析失败: {str(e)}", "title": "", "description": ""}

    try:
        result = json.loads(text)
        if "title" not in result:
            result["title"] = ""
        if "description" not in result:
            result["description"] = ""
        return result
    except json.JSONDecodeError:
        fallback_description = (
            text.strip().replace("```json", "").replace("```", "").strip()[:50]
        )
        return {
            "error": "无法提取JSON内容",
            "title": "",
            "description": fallback_description,
        }



async def image_to_base64_async(image_path: str) -> str:
    """
    异步将图像文件转换为base64编码字符串
    参数:
        image_path: 图像文件路径
    返回:
        base64编码的图像字符串
    """
    try:
        async with aiofiles.open(image_path, "rb") as image_file:
            image_data = await image_file.read()
            encoded_string = base64.b64encode(image_data).decode("utf-8")
        return encoded_string
    except FileNotFoundError:
        raise FileNotFoundError(f"文件未找到: {image_path}")
