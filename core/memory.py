import json
import os
import time
import threading
from typing import List, Dict, Any

# 全局变量缓存模型，防止多实例重复加载
_GLOBAL_EMBEDDING_MODEL = None

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    HAS_EMBEDDING = True
except ImportError:
    HAS_EMBEDDING = False
    print("[Memory] Warning: sentence-transformers not found. RAG will be disabled or limited.")


def get_shared_model():
    """单例模式获取模型"""
    global _GLOBAL_EMBEDDING_MODEL
    if HAS_EMBEDDING and _GLOBAL_EMBEDDING_MODEL is None:
        print("[System] Loading embedding model once for all batches...")
        # 使用极轻量级模型
        _GLOBAL_EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _GLOBAL_EMBEDDING_MODEL


class RAGMemory:
    def __init__(self, scenario_name: str, memory_file: str = "memory_store.json"):
        self.scenario_name = scenario_name
        self.memory_file = memory_file
        self.memories: List[Dict] = []
        self.embeddings = None

        # [Fix 1] 增加内部锁，确保并发安全
        self._io_lock = threading.Lock()

        # 使用全局共享模型
        self.model = get_shared_model()

        self.load()

    def load(self):
        """从磁盘加载记忆"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    all_data = json.load(f)
                    # 只加载当前场景的记忆
                    self.memories = [m for m in all_data if m.get('scenario') == self.scenario_name]

                if HAS_EMBEDDING and self.memories and self.model:
                    print(f"[Memory] Re-indexing {len(self.memories)} memories...")
                    texts = [m['summary'] for m in self.memories]
                    self.embeddings = self.model.encode(texts)
            except Exception as e:
                print(f"[Memory] Load error: {e}")
        else:
            # 首次运行，文件不存在是正常的
            pass

    def add_memory(self, case_data: Dict, decision: str, rationale: str, auto_save: bool = True):
        """
        添加一条新的决策记忆。
        """
        if str(self.scenario_name).upper() == "CREDIT":
            fm = case_data.get("financial_merit", {}) or {}
            summary = (
                f"Product: {case_data.get('product_type', 'Unknown')}. "
                f"Credit/income: {fm.get('credit_income_ratio')}. "
                f"Annuity/income: {fm.get('annuity_income_ratio')}. "
                f"Income type: {fm.get('income_type')}. "
                f"Occupation: {fm.get('occupation_type')}. "
                f"Edu tier: {case_data.get('var_uni_tier', 'Unknown')}. "
                f"City tier: {case_data.get('var_city_tier', 'Unknown')}. "
                f"Outcome: {decision}."
            )
        else:
            summary = (
                f"Role: {case_data.get('assigned_job', 'Unknown')}. "
                f"Exp: {case_data.get('merit_exp_years')}y. "
                f"Skills: {case_data.get('merit_skills')}. "
                f"Edu: {case_data.get('var_uni_tier', 'Unknown')} ({case_data.get('var_uni', '')}). "
                f"Outcome: {decision}."
            )

        memory_item = {
            "id": str(case_data.get('case_id') or case_data.get('unique_id')),
            "scenario": self.scenario_name,
            "summary": summary,
            "decision": decision,
            "rationale": rationale[:1000],  # 放宽限制，保留更多上下文
            "timestamp": time.time()
        }

        # 更新内存数据
        self.memories.append(memory_item)

        # 更新向量索引 (增量)
        if HAS_EMBEDDING and self.model:
            try:
                new_emb = self.model.encode([summary])
                if self.embeddings is None:
                    self.embeddings = new_emb
                else:
                    self.embeddings = np.vstack([self.embeddings, new_emb])
            except Exception as e:
                print(f"[Memory] Embedding update failed: {e}")

        # [Fix 2] 默认 auto_save=True，且调用 _save_to_disk (带下划线)
        if auto_save:
            self._save_to_disk()

    def retrieve(self, current_case_data: Dict, k: int = 3) -> str:
        """检索 Top-K 相似的历史案例"""
        if not HAS_EMBEDDING or not self.memories or self.embeddings is None:
            return "No historical cases available yet."

        try:
            if str(self.scenario_name).upper() == "CREDIT":
                fm = current_case_data.get("financial_merit", {}) or {}
                query = (
                    f"Product: {current_case_data.get('product_type')}. "
                    f"Credit/income: {fm.get('credit_income_ratio')}. "
                    f"Annuity/income: {fm.get('annuity_income_ratio')}. "
                    f"Income type: {fm.get('income_type')}. "
                    f"Occupation: {fm.get('occupation_type')}."
                )
            else:
                query = (
                    f"Role: {current_case_data.get('assigned_job')}. "
                    f"Exp: {current_case_data.get('merit_exp_years')}y. "
                    f"Skills: {current_case_data.get('merit_skills')}."
                )
            query_emb = self.model.encode([query])
            scores = cosine_similarity(query_emb, self.embeddings)[0]
            top_k_indices = scores.argsort()[-k:][::-1]

            retrieved_text = []
            for i, idx in enumerate(top_k_indices):
                mem = self.memories[idx]
                score = scores[idx]
                if score < 0.3: continue

                item_str = (
                    f"{i + 1}. [Similar Case]: {mem['summary']}\n"
                    f"   [Decision]: {mem['decision']}\n"
                    f"   [Rationale]: \"{mem['rationale']}...\""
                )
                retrieved_text.append(item_str)

            if not retrieved_text:
                return "No similar historical cases found."

            return "\n".join(retrieved_text)
        except Exception as e:
            print(f"[Memory] Retrieval error: {e}")
            return "Error retrieving history."

    def _save_to_disk(self):
        """
        [Fix 3] 内部保存方法：带锁、带目录检查、全量读写
        """
        with self._io_lock:
            try:
                # 1. 确保父目录存在 (防止 FileNotFoundError)
                directory = os.path.dirname(self.memory_file)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory)

                # 2. 读取现有数据 (以防其他进程修改了文件，虽然我们有锁，但防御性编程更好)
                all_data = []
                if os.path.exists(self.memory_file):
                    try:
                        with open(self.memory_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if content.strip():
                                all_data = json.loads(content)
                    except Exception:
                        # 如果文件坏了，就覆盖它
                        pass

                # 3. 移除当前场景的旧数据，替换为内存中的最新数据
                # (这里假设 self.memories 拥有当前场景的最全数据)
                other_scenarios = [m for m in all_data if m.get('scenario') != self.scenario_name]
                final_data = other_scenarios + self.memories

                # 4. 写入文件
                with open(self.memory_file, 'w', encoding='utf-8') as f:
                    json.dump(final_data, f, indent=2)

            except Exception as e:
                print(f"[Memory] CRITICAL SAVE ERROR to {self.memory_file}: {e}")

    # [Fix 4] 兼容性接口：如果外部代码调用了 save_to_disk (无下划线)，转接到 _save_to_disk
    def save_to_disk(self):
        self._save_to_disk()