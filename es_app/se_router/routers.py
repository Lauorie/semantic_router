# pip insall semantic-router[local]

from es_app.se_router.utterances import Utterances
from semantic_router import Route
from semantic_router.layer import RouteLayer
from semantic_router.encoders import HuggingFaceEncoder

class SemanticsRouter:
    def __init__(self, encoder_name='/root/app/models/models--iampanda--zpoint_large_embedding_zh'):
        self.routes = self._get_routes_from_utterances()
        self.encoder = HuggingFaceEncoder(
            name=encoder_name,
            tokenizer_kwargs={"max_length": 512, "truncation": True}
        )
        self.rl = RouteLayer(encoder=self.encoder, routes=self.routes)
    
    def _get_routes_from_utterances(self):
        return [getattr(Utterances, attr) for attr in dir(Utterances) if isinstance(getattr(Utterances, attr), Route)]
    
    def route(self, query: str):
        return self.rl(query).name

# 使用示例
if __name__ == "__main__":
    router = SemanticsRouter()
    result = router.route("结果以表格形式展示")
    print(result)
    # Output: table