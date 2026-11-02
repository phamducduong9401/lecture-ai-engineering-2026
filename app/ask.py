"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 既定の質問の後、続けて質問できる

ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors
from google.genai import types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask_with_history(
    client: genai.Client,
    history: list[types.Content],
    question: str,
) -> str:
    history.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))
    response = client.models.generate_content(model=MODEL, contents=history)

    usage = response.usage_metadata
    print(f"[入力トークン数] {usage.prompt_token_count}")
    print(f"[出力トークン数] {usage.candidates_token_count}\n")

    if not response.candidates or response.candidates[0].content is None:
        raise RuntimeError("モデルから回答を受け取れませんでした。")
    history.append(response.candidates[0].content)
    return response.text


def ask(question: str) -> str:
    """質問を1回実行する（会話履歴なし）。"""
    client = genai.Client(api_key=get_api_key())
    return ask_with_history(client, [], question)


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        client = genai.Client(api_key=get_api_key())
        history: list[types.Content] = []
        question = args.question
        while question:
            print(f"[質問] {question}\n")
            print(ask_with_history(client, history, question))
            try:
                question = input("\n続けて質問してください（終了するには空行）: ").strip()
            except EOFError:
                break
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
