import argparse
import asyncio
import re
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from llm_client import LLMClient
from preprocess import load_data


categories_from_indices = {
    0: 'нет конкретного ответа',
    1: 'ок',
    2: 'work-life balance',
    3: 'адекватные планы и количество метрик',
    4: 'бесплатное питание',
    5: 'бюрократия',
    6: 'взаимодействие',
    7: 'внерабочие активности',
    8: 'график работы',
    9: 'дополнительные сотрудники',
    10: 'идея по продукту',
    11: 'карьерный рост',
    12: 'клиенты',
    13: 'конкурсы',
    14: 'культура обратной связи',
    15: 'лояльность к сотрудникам',
    16: 'льготы',
    17: 'мерч',
    18: 'нездоровая атмосфера',
    19: 'обучение',
    20: 'оплата труда',
    21: 'офисное пространство',
    22: 'подарки на праздники',
    23: 'премии',
    24: 'процессы',
    25: 'сложность работы',
    26: 'техника/ит',
    27: 'удаленная работа',
    28: 'оплата сверхурочного труда',
    29: 'руководитель',
}


def extract_int(value: str) -> Optional[int]:
    try:
        value = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL)
        return int(value.strip())
    except (ValueError, AttributeError):
        return None


async def classify_comment(comment: str, system_prompt: str, client: LLMClient) -> tuple[Optional[int], Optional[str]]:
    reply, response = await client.chat(
        text=comment,
        system_prompt=system_prompt,
        temperature=0.01,
        max_tokens=10000,
    )

    idx = extract_int(reply)
    if idx is None:
        return None, None, reply, response

    category = categories_from_indices.get(idx)
    if category:
        return idx, category, reply, response
    return None, None, reply, response


async def classify_all(texts, system_prompt: str, client: LLMClient):
    tasks = [classify_comment(t, system_prompt, client) for t in texts]
    return await asyncio.gather(*tasks)


async def main():
    parser = argparse.ArgumentParser(description="Classify eNPS survey comments using DeepSeek LLM.")
    parser.add_argument("--input", help="Path to input Excel or CSV file (columns: Score, A1, C1, A2, C2)")
    parser.add_argument("--output", help="Path to output Excel file (default: <input_stem>_classified.xlsx)")
    parser.add_argument("--prompt", default="prompt.txt", help="Path to system prompt file (default: prompt.txt)")
    parser.add_argument("--model", default="deepseek-chat", help="Model name to use (default: deepseek-chat)")
    parser.add_argument("--thinking", choices=["true", "false"], default=None)
    args = parser.parse_args()

    thinking = {"true": True, "false": False}.get(args.thinking)
    client = LLMClient(model=args.model, thinking=thinking)
    system_prompt = Path(args.prompt).read_text(encoding="utf-8")

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_classified.xlsx")

    df = load_data(str(input_path))

    texts = df["A"].tolist()
    pred_idx, pred_cat, replies, responses = [], [], [], []

    for i in tqdm(range(0, len(texts), 100)):
        results = await classify_all(texts[i:i + 100], system_prompt, client)
        for idx, cat, reply, response in results:
            pred_idx.append(idx)
            pred_cat.append(cat)
            replies.append(reply)
            responses.append(response)
        # await asyncio.sleep(4)

    df["pred_idx"] = pred_idx
    df["pred_category"] = pred_cat
    df["model_reply"] = replies
    df["model_response"] = responses
    df.to_excel(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
