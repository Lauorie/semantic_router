import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

from es_app.se_router.routers import SemanticsRouter

class Agent:
    def __init__(self):
        self.router = SemanticsRouter()

    def route(self, query: str):
        return self.router.route(query)
    
    def process(self, query: str):
        pass

if __name__ == "__main__":
    agent = Agent()
    print(agent.route("Please summarize our conversation"))
    print(agent.route("翻译小型指令的第五章"))
    print(agent.route("请将对话用表格形式展示"))
    