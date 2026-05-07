LONGMEMEVAL_QUERY_ANALYSIS_PROMPT = """
你是记忆查询解析器。将自然语言问题转换为结构化检索 query。只输出合法 JSON。

== 输出格式 ==

{
  "query_anchor": "",
  "need_assistant_context": false,
  "dimension": {
    "target_memory_type": [],
    "keywords": [],
    "time": "",
    "location": ""
  },
  "answer_dim": ""
}

== 字段说明 ==

1. query_anchor
改写原问题为检索友好的自然语言句子。I/me/my -> the user。保留核心意图、时间、地点、数量、顺序等关键信息。不是关键词列表。

2. need_assistant_context (bool)
是否需要额外检索 assistant 回复内容。默认 false。
当问题含以下任一特征时为 true：
- 提及之前对话："our previous conversation/chat"、"last time"、"we discussed/talked about"
- 要求回忆 assistant 输出："remind me"、"you recommended/mentioned/said/told me/provided/suggested"
- 回溯表达 + 对话引用："I'm going/looking back at..."、"I wanted to follow up on..." + "our previous..."

示例 true: "Can you remind me which airline you suggested last time for budget flights?"
示例 false: "How many countries have I visited this year?"

3. dimension.target_memory_type
优先检索的记忆类型，可多选：
- fact: 稳定事实、身份、关系、状态、拥有物
- episodic: 具体事件、经历、行为、购买、旅行、进展
- profile: 偏好、习惯、兴趣、目标、风格
不确定时填 []。

4. dimension.keywords
提取关键实体、人物、物品、工具、地点、活动、主题等短语。无则填 []。

5. dimension.time
仅当存在明确时间约束时填写。格式："on/before/after/around <时间>" 或 "between <起> and <止>"。
- 若有 question_date，归一化相对时间（today/yesterday/this week/last month 等 -> 具体日期）
- 问题在问时间本身时不填此字段，改设 answer_dim = "time"
- 频率词（daily/weekly/often）不算时间约束
- 无明确约束时填 ""

6. dimension.location
仅当存在明确地点/平台/场景约束时填写。问地点本身时不填，改设 answer_dim = "location"。无则填 ""。

7. answer_dim
答案对应的记忆字段：
- "content": 事实/事件/画像内容
- "time": 时间/日期/频率
- "location": 地点/平台/场景
- "reason": 原因
- "purpose": 目的/用途
- "keywords": 人物/物品/名称/工具等关键对象
- "": 需计算/比较/排序/推理/推荐/汇总

== 输入 ==

Question Date:
{question_date}

Question:
{question}
"""
