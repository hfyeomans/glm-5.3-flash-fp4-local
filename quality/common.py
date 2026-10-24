"""Small auditable pieces shared by the evaluation runner and report."""
import hashlib
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def visible_answer(raw):
    # The shared prompt ends in <think>; do not grade private reasoning as an answer.
    if '</think>' not in raw:
        return '', False
    return raw.split('</think>', 1)[1].strip(), True


def wilson(correct, total, z=1.959963984540054):
    if total == 0:
        raise ValueError('No scored samples')
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0, center - half), min(1, center + half)]
