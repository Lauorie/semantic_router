# 当前是取每一类下 mean 值，结合阈值判断
# 还有一种解法 → 用 reranker model 排序以后取 top1 的 name，当然也要结合 threshold


import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from embeddings import EmbeddingsClient
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Route:
    name: str
    utterances: List[str]
    
    def __post_init__(self):
        self.embedding_model = EmbeddingsClient()
        self.average_embedding = self.calculate_average_embedding()

    def calculate_average_embedding(self) -> np.ndarray:
        try:
            embeddings = [self.embedding_model.encode(u) for u in self.utterances]
            return np.mean(embeddings, axis=0)
        except Exception as e:
            print(f"Error calculating embedding for route {self.name}: {e}")
            return np.zeros(1792)

class RouteMatcher:
    def __init__(self, routes: List[Route], threshold: float = 0.7):
        self.routes = routes
        self.threshold = threshold
        self.embedding_model = EmbeddingsClient()
        self.route_embeddings = np.array([route.average_embedding for route in self.routes])
        self.route_names = [route.name for route in self.routes]

    def match_query(self, query: str) -> str:
        try:
            query_embedding = self.embedding_model.encode(query)
            similarities = cosine_similarity([query_embedding], self.route_embeddings)[0]
            best_match_index = np.argmax(similarities)
            best_match_similarity = similarities[best_match_index]
            
            if best_match_similarity < self.threshold:
                return "rag"
            return self.route_names[best_match_index]
        except Exception as e:
            print(f"Error matching query: {e}")
            return "rag"

# 定义数据类
class Utterances:
    translation = Route(
        name="translation",
        utterances=[
            "翻译我们的对话信息",
            "请你翻译一下刚刚的对话",
            "翻译之前的对话",
            "翻译对话",
            "翻译我们的对话内容",
            "将对话内容翻译成",
            "请转换我们的对话为",
            "将对话转换为",
            "转换对话内容为",
            "请求翻译最近的对话",
            "把对话翻译成",
            "请翻译对话内容",
            "用另一种语言解释对话",
            "将对话翻译成其他语言",
            "将对话翻译为",
            "翻译上述内容",
            "把摘要翻译成英语",
            "请将对话翻译成英文",
        ]
    )

    summary = Route(
        name="summary",
        utterances=[
            "请你总结我们的对话信息",
            "总结一下刚才的对话",
            "总结之前的对话",
            "总结对话",
            "请简述我们的对话",
            "对话内容简要说明",
            "对话的摘要",
            "请提供对话的概览",
            "对话的快速总结",
            "简要描述我们的对话",
            "对话的重点",
            "请概括对话",
            "对话的主要观点",
            "对话的精髓",
        ]
    )

    table = Route(
        name="table",
        utterances=[
            "请你将我们的对话信息整理成表格",
            "把刚才的对话整理成表格",
            "将之前的对话信息整理成表格",
            "整理对话成表格",
            "对话内容整理成表格形式",
            "请将数据整理成表格",
            "对话的表格形式",
            "整理对话内容为表格",
            "对话信息的表格版",
            "将对话信息转化为表格",
            "请制作对话的表格",
            "对话的表格呈现",
            "以表格形式展现对话",
            "将对话整理为结构化表格",
            "用表格表示上述结果",
            "使用表格整理上述结果",
            "请将对话整理成表格形式",
            "请用表格的形式列举出来",
        ]
    )

routes = [
    Utterances.translation,
    Utterances.summary,
    Utterances.table
]

matcher = RouteMatcher(routes, threshold=0.7)  # 设置阈值为0.7

# 测试
test_queries = [
    "能不能把我们的对话翻译成英文？",
    "请总结一下我们刚才讨论的内容",
    "可以把这些信息整理成一个表格吗？",
    "今天天气怎么样？",
    "天空为什么是蓝色的？",
]

for query in test_queries:
    matched_route = matcher.match_query(query)
    print(f"Query: {query}")
    print(f"Matched Route: {matched_route}\n")