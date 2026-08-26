import autogen
import re
import json
import copy
from typing import Dict


class PrivacyPreservingGroupChat(autogen.GroupChat):
    """
    带隐私保护的群聊组件。
    功能：
    1. 拦截 PRIVATE_INTERNAL_THOUGHT，防止泄露给其他 Agent。
       支持两种输出格式：
         - JSON 格式（当前标准）：直接操作 JSON 对象，将字段替换为占位符
         - numbered 格式（旧版兼容）：正则移除 "0. PRIVATE_INTERNAL_THOUGHT: ..."
    2. 将完整记录存入 full_transcript 用于科研分析（研究者可见完整内容）。

    设计原则：
      PRIVATE_INTERNAL_THOUGHT 是 Agent 的私有空间，对其他 Agent 不可见，
      但对研究者（通过 full_transcript）完全可见，用于偏见探测分析。
    """

    def __init__(self, agents, messages, max_round=10, speaker_selection_method="auto",
                 allow_repeat_speaker=True, **kwargs):
        super().__init__(
            agents=agents,
            messages=messages,
            max_round=max_round,
            speaker_selection_method=speaker_selection_method,
            allow_repeat_speaker=allow_repeat_speaker,
            **kwargs
        )
        self.full_transcript = []

    def append(self, message: Dict, speaker: autogen.Agent):
        # 1. Deepcopy 并存储完整记录（研究者可见，用于偏见分析）
        full_msg = copy.deepcopy(message)
        self.full_transcript.append(full_msg)

        # 2. 清洗内容（传给其他 Agent 时隐藏私有字段）
        content = message.get("content", "")
        if content:
            sanitized = self._sanitize(content)
            message["content"] = sanitized

        super().append(message, speaker)

    def _sanitize(self, content: str) -> str:
        """
        移除 PRIVATE_INTERNAL_THOUGHT 和 PRIVATE_KEYWORDS，
        保留所有 PUBLIC_* 字段和 DECISION_SIGNAL。

        优先尝试 JSON 解析；失败则回退到正则（兼容旧版 numbered 格式）。
        """
        stripped = content.strip()

        # ── JSON 格式处理 ──────────────────────────────────────────────────────
        # 提取 JSON 块（可能有前缀/后缀文本）
        start = stripped.find("{")
        end   = stripped.rfind("}") + 1
        if start != -1 and end > start:
            json_str = stripped[start:end]
            try:
                data = json.loads(json_str)
                # 替换私有字段为占位符（不删除键，保持 JSON 结构完整）
                if "PRIVATE_INTERNAL_THOUGHT" in data:
                    data["PRIVATE_INTERNAL_THOUGHT"] = "[HIDDEN]"
                if "PRIVATE_KEYWORDS" in data:
                    data["PRIVATE_KEYWORDS"] = ["[HIDDEN]"]
                sanitized_json = json.dumps(data, ensure_ascii=False)
                # 重新拼接（保留 JSON 块前后的非 JSON 文本）
                return stripped[:start] + sanitized_json + stripped[end:]
            except (json.JSONDecodeError, Exception):
                pass  # JSON 解析失败，回退到正则

        # ── 正则格式处理（旧版 numbered 输出兼容）──────────────────────────────
        pattern = r"(0\.\s*PRIVATE_INTERNAL_THOUGHT:.*?)(?=\n\d\.\s*[A-Z_]+:|$)"
        sanitized = re.sub(
            pattern,
            "[SYSTEM: Internal Thought Hidden]",
            stripped,
            flags=re.DOTALL | re.IGNORECASE
        )
        return sanitized.strip()
