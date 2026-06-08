"""
四元组格式化模块
==================
KG中的四元组格式 (s, r, o, t)

功能:
1. 统一多种数据源为四元组格式
2. 四元组验证与清洗 (去除无效时间戳、空实体)
3. 导出为标准格式 (CSV, TTL, 内部表示)
"""

from typing import List, Tuple, Set, Optional
import csv
import json


def validate_quadruple(quad: Tuple[str, str, str, str]) -> bool:
    """
    验证四元组的合法性
    论文公式(1): 规则格式验证
    """
    s, r, o, t = quad
    if not all([s, r, o, t]):
        return False
    if s.strip() == o.strip():
        return False  # 自环通常不合法
    return True


def clean_quadruples(quads: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str, str, str]]:
    """
    清洗四元组: 去除无效、重复、空值的数据
    """
    seen = set()
    cleaned = []
    for q in quads:
        if not validate_quadruple(q):
            continue
        key = (q[0].strip(), q[1].strip(), q[2].strip(), q[3].strip())
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(key)
    return cleaned


def format_quad_for_llm(quad: Tuple[str, str, str, str],
                         include_time: bool = True) -> str:
    """
    将四元组格式化为LLM可理解的自然语言描述
    Example:
        (Barack_Obama, Make_a_visit, China, 2014-12-09) →
        "Barack Obama made a visit to China on 2014-12-09"
    """
    s, r, o, t = quad
    s_readable = s.replace("_", " ")
    o_readable = o.replace("_", " ")
    r_readable = r.replace("_", " ").lower()

    if include_time:
        return f"{s_readable} {r_readable} {o_readable} on {t}"
    else:
        return f"{s_readable} {r_readable} {o_readable}"


def quadruples_to_rdf(quads: List[Tuple[str, str, str, str]],
                       namespace: str = "http://tkg.lpehd.org/") -> str:
    """
    将四元组转换为RDF/TTL格式
    供SPARQL查询使用 (论文SPARQL事件模式匹配)
    """
    lines = [
        "@prefix tkg: <http://tkg.lpehd.org/ontology/> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
    ]
    for i, (s, r, o, t) in enumerate(quads):
        s_clean = s.replace(" ", "_").replace("'", "_")
        r_clean = r.replace(" ", "_").replace("'", "_")
        o_clean = o.replace(" ", "_").replace("'", "_")
        lines.append(f"tkg:quadruple_{i} tkg:subject tkg:{s_clean} ;")
        lines.append(f"    tkg:relation tkg:{r_clean} ;")
        lines.append(f"    tkg:object tkg:{o_clean} ;")
        lines.append(f"    tkg:timestamp \"{t}\"^^xsd:date .")
        lines.append("")
    return "\n".join(lines)


def quadruples_to_json(quads: List[Tuple[str, str, str, str]], path: str):
    """导出四元组为JSON格式"""
    data = [
        {"subject": s, "relation": r, "object": o, "timestamp": t}
        for s, r, o, t in quads
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
