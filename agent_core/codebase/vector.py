import math


def cosine_similarity(
    left: list[float],
    right: list[float],
) -> float:
    if len(left) != len(right):
        raise ValueError("向量维度必须一致")

    if not left:
        raise ValueError("向量不能为空")

    dot_product = sum(
        a * b
        for a, b in zip(
            left,
            right,
        )
    )

    left_norm = math.sqrt(sum(value * value for value in left))

    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)
