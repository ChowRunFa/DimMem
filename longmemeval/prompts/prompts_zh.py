LONGMEMEVAL_STRUCTURED_MEMORY_EXTRACTION_PROMPT = """
你是一个结构化记忆抽取器。

任务：从输入的用户—助手对话记录中抽取有长期价值的结构化记忆，并严格输出合法 JSON。
只能输出 JSON，不要输出解释、分析、Markdown 或额外文本。

========================
输出格式
========================

{
  "memories": [
    {
      "source_id": 1,
      "content": "",
      "dimension": {
        "memory_type": "",
        "time": "",
        "location": "",
        "reason": "",
        "purpose": "",
        "keywords": []
      }
    }
  ]
}

========================
抽取目标
========================

应抽取：
1. 事实信息、身份背景、关系、当前状态、工具、模型、数据集、项目配置；
2. 具体经历、事件、行为、阶段进展、未来计划；
3. 长期偏好、习惯、兴趣、价值观、目标、能力、交互或写作风格；
4. 对未来理解用户需求、检索用户背景有帮助的信息。

不抽取：
1. 问候、感谢、简单确认、无意义闲聊；
2. 临时格式要求、一次性操作指令、缺乏长期价值的当前任务细节。

========================
memory_type
========================

dimension.memory_type 只能是 fact、episodic、profile。

fact：稳定事实，回答"是什么 / 有什么 / 使用什么 / 关系是什么 / 当前状态是什么"。
包括身份、背景、关系、状态、工具、模型、数据集、配置、已确定选择、稳定属性。
例：The user uses LLaMA2-7B as the base model.

episodic：具体事件，回答"发生了什么 / 做了什么 / 经历了什么 / 计划做什么"。
包括一次事件、经历、行动、阶段进展、未来计划，以及有时间/地点/场景的具体事实。
例：The user plans to train a local LLaMA2-7B model using Urdu data.

profile：长期画像，回答"长期是什么样 / 喜欢什么 / 通常做什么 / 相信什么"。
包括偏好、习惯、兴趣、价值观、长期目标、能力特征、风格偏好、稳定行为模式。
例：The user prefers concise Java code with a single main function.

无法归入 fact、episodic 或 profile 的内容不要抽取。

========================
content 规则
========================

content 是记忆的主文本，必须是一句完整、自包含、可检索的话。

要求：
1. 明确说明这条记忆关于谁，以及核心事实、事件或画像内容。
2. 如果原文包含时间、地点、原因、目的，应尽量写入 content。
3. 消除模糊代词，使 content 不依赖原始上下文。
4. 相对时间应根据消息时间戳归一化，例如 yesterday → 具体日期。
5. 不要加入原文没有支持的信息，也不要把单次事件过度概括成长期画像。

========================
dimension 规则
========================

dimension 用于结构化检索。除 memory_type 外，无明确依据则填 ""；keywords 无明确关键词则填 []。

time：记忆成立、发生、计划发生或重复出现的时间。
有绝对日期用绝对日期；有相对时间且有消息时间戳时必须归一化；无时间填 ""。
不要使用当前系统时间，除非它就是消息时间戳。

location：物理地点、线上平台、组织场景、家庭空间、工作场所、系统环境或活动场所。
只有原文明确提到或强烈暗示时填写；不要把普通主题强行写成 location。

reason：原因、动因、触发因素或背景条件。
只有原文明确说明或强烈暗示时填写；不要推测隐藏动机；不要和 purpose 混淆。

purpose：目标、意图或期望结果。
只有原文明确说明或强烈暗示时填写；不要推测未说明的目的。

keywords：用于检索、去重和 query-memory 对齐的关键词或短语。
应提取主体、对象、工具、模型、数据集、项目、人物、地点、活动、结果、偏好对象、兴趣领域等。
必须是简短词语或名词短语；不要放完整句子；不要重复；不要提取无检索价值的普通词。

========================
抽取规则
========================

1. 按消息顺序处理。
2. 主要从用户消息中抽取。
3. 每条 memory 应尽量原子化。若一条消息较长或包含多个独立信息点，应拆分为多条 memory，分别保留其中的人物、时间、地点、事件、原因、目的、偏好、物品等关键细节，避免合并过度或遗漏。
4. content 必须自包含，不能依赖原始对话上下文。
5. time 应根据消息时间戳归一化：相对时间表达需转换为绝对日期或时间范围；若消息时间为 2023-05-08，原文说 yesterday，则 time 写 2023-05-07。
6. 简单确认、临时格式要求、当前一次性任务一般不抽取。
7. 输出必须是合法 JSON，不要输出 JSON 以外的任何文本。

{overlap_rule}

以下是你现在需要处理的真实输入：

{conversation}
"""


OVERLAP_RULE = '''
========================
输入与重叠上下文规则
========================

你将收到一个"当前对话片段"。
输入中前 {overlap_count} 条消息，即编号 1 到 {overlap_count}，是上一段的重叠上下文，不能作为新的记忆来源。
前 {overlap_count} 条只用于理解后续消息；真正允许抽取的内容从第 {extract_start_index} 条消息开始。
'''

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


LONGMEMEVAL_OFFLINE_UPDATE_PROMPT = """

"""



__all__ = [
    "LONGMEMEVAL_OFFLINE_UPDATE_PROMPT",
    "LONGMEMEVAL_QUERY_ANALYSIS_PROMPT",
    "LONGMEMEVAL_STRUCTURED_MEMORY_EXTRACTION_PROMPT",
    "OVERLAP_RULE",
]