import asyncio
import sys
from typing import Optional

from llm_client import llm_client


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
        return int(value.strip())
    except (ValueError, AttributeError):
        return None


async def classify_comment(comment: str) -> None:
    reply = await llm_client.chat(
        text=comment,
        temperature=0.0,
        max_tokens=1000,
    )

    idx = extract_int(reply)
    if idx is None:
        return

    category = categories_from_indices.get(idx)
    if category:
        print(category)


def main():
    if sys.stdin.isatty():
        comment = input("Введите комментарий: ").strip()
    else:
        comment = sys.stdin.read().strip()

    if not comment:
        return

    asyncio.run(classify_comment(comment))


if __name__ == "__main__":
    main()
