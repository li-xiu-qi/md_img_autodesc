# Markdown Image Auto-Description

为 Markdown 文件中的本地图片自动生成 `alt` 和 `title` 描述。

## 功能

- 提取 Markdown 文件中的所有本地图片链接。
- 使用 OpenAI GPT-4o 多模态模型分析图片内容。
- 自动为图片生成描述性的 `alt` 文本和 `title` 标题。
- 将生成的描述更新回 Markdown 文件中，同时保留原始图片链接。

## 安装

1. 克隆或下载本项目。

2. 确保你已安装 Python 3.7+。

3. 建议在虚拟环境中安装依赖：

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4. 安装所需的库：

    ```bash
    pip install -r requirements.txt
    ```

## 配置

1. 在项目根目录创建一个名为 `.env` 的文件。

2. 在 `.env` 文件中添加你的 **智谱API密钥**（推荐）或 OpenAI API 密钥，以及自定义基础 URL（如有需要）：

    **智谱（推荐，默认使用 flash 模型）：**

    ```env
    ZHIPU_API_KEY="你的智谱API密钥"
    # 可选，基础URL，通常无需更改
    ZHIPU_BASE_URL="https://open.bigmodel.cn/api/paas/v4/"
    ZHIPU_VISION_MODEL="glm-4v-flash"  # 默认已自动使用 flash，无需手动设置
    ```

    **如需使用 OpenAI，可配置如下：**

    ```env
    OPENAI_API_KEY="sk-your-key-here"
    # 可选，如果使用代理或第三方服务
    OPENAI_BASE_URL="https://api.openai.com/v1"
    ```

## 使用方法


使用 `main.py` 脚本来处理你的 Markdown 文件，默认使用智谱 glm-4v-flash 多模态模型。


```bash
python main.py -i [输入文件路径] [-o 输出文件路径] [-p 服务商] [-m 模型名] [-t 标题最小字数] [-d 描述最小字数]
```


**参数说明:**

- `-i, --input`: 必需参数，指定要处理的原始 Markdown 文件的路径。
- `-o, --output`: 可选参数，指定处理后新文件的保存路径。
    - 如果省略此参数，脚本会自动在原文件名基础上添加 `_with_desc` 后缀生成新文件，例如 `README.md` 会生成 `README_with_desc.md`。
- `-p, --provider`: 可选，指定服务商（zhipu 或 openai），默认 zhipu。
- `-m, --model`: 可选，指定多模态模型名，默认使用智谱 glm-4v-flash。
- `-t, --title-min-length`: 可选，指定图片标题（title）的最小字数，默认为 10。
- `-d, --desc-min-length`: 可选，指定图片描述（alt）的最小字数，默认为 50。

**示例:**

1. **处理文件并自动生成带描述的新文件（推荐）：**

    ```bash
    python main.py 量化.md
    # 输出文件为 量化_with_desc.md
    ```

2. **处理文件并自定义输出文件名：**

    ```bash
    python main.py 量化.md -o output.md
    ```

3. **使用 uv 运行（推荐 uv 用户）：**

    ```bash
    uv run main.py 量化.md
    # 输出文件为 量化_with_desc.md
    ```

4. **如需使用 OpenAI，可加参数：**

    ```bash
    python main.py my_document.md --provider openai --model gpt-4o
    # 输出文件为 my_document_with_desc.md
    ```
