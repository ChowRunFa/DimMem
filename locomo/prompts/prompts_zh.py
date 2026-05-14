from cgi import MiniFieldStorage


LOCOMO_STRUCTURED_MEMORY_EXTRACTION_PROMPT = f"""
你是一个结构化记忆抽取器。

你的任务是：根据输入的多人对话记录，抽取有长期价值的结构化记忆，并严格输出合法 JSON。

你只能输出 JSON，不要输出解释、分析、Markdown 或额外文本。

========================
一、输出格式
========================

只返回如下 JSON 对象：

{
  "memories": [
    {
      "source_id": 1,
      "source_speaker": "",
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
二、字段说明
========================

1. source_id

表示该条记忆主要来源于输入对话中的哪一条消息编号。

2. source_speaker

表示 source_id 对应消息的说话人。

3. content

content 是记忆的核心文本表示，必须完整、自包含，可独立理解和检索。

核心要求有：

- 指代消解：明确说明主体是谁、核心信息是什么，不使用“他”“她”“那里”“这个”等模糊指代，根据上下文还原具体人物、对象、地点或事件。

- 关键信息保留：尽量保留原文中的关键要素：时间、地点、原因、目的、对象、人物关系、媒介证据等，不添加原文没有支持的信息。

- 时间归一化：content 中涉及时间时，应基于消息时间戳归一化
  例如：yesterday → 具体日期；last week → 日期范围；next month → 具体月份；three years ago → 具体或近似年份；

4. dimension.memory_type

memory_type 只能是以下三类之一：fact、episodic、profile。

fact：
用于记录稳定事实、身份、背景、关系、当前状态、工具、模型、数据集、配置、拥有物或稳定属性。

判断标准：
如果信息主要回答“是什么 / 有什么 / 使用什么 / 关系是什么 / 当前状态是什么”，标为 fact。

示例：
- Jack has two children.
- Mico is Jack's close friend.

episodic：
用于记录具体事件、经历、行为、购买、旅行、见面、分享、计划、阶段进展或时间线相关信息。

判断标准：
如果信息主要回答“发生了什么 / 做了什么 / 经历了什么 / 计划做什么 / 什么时候发生 / 在哪里发生”，标为 episodic。

注意：
- 照片、图片、视频、截图、语音等媒介，如果承载了某次经历、见面、活动、旅行、家庭事件、朋友互动或重要对象，也应抽取为 episodic；
- 不要因为信息是“拍照”“发图”“分享照片”就忽略，只要它指向可检索的具体经历，就应保留。

示例：
- Nora volunteered at an animal shelter on 2023-07-11.
- Ethan joined a river cleanup with his sister on 2023-07-30.
- Maria shared a photo from a family dinner in May 2023.

profile：
用于记录长期偏好、习惯、兴趣、价值观、长期目标、能力特征、交互偏好、稳定行为模式或长期支持来源。

判断标准：
如果信息主要回答“长期是什么样 / 喜欢什么 / 通常做什么 / 相信什么 / 重视什么 / 依靠什么 / 被什么激励”，标为 profile。

注意：
- 朋友、家人、导师、伴侣、同事、宠物等，如果被描述为重要支持、动力来源、长期陪伴或稳定关系，应作为 profile 或 fact 抽取；
- 如果只是一次见面或一次互动，标为 episodic；
- 如果表达长期关系或稳定影响，标为 profile 或 fact。

示例：
- Maria practices yoga every Saturday morning.
- Mico believes every child deserves love, acceptance, and a safe home.
- Ethan sees his older sister as an important source of support.

无法归入 fact、episodic 或 profile 的内容，不要抽取。

5. dimension.time

表示记忆成立、发生、计划发生或重复出现的时间。

规则：
- 有绝对日期则使用绝对日期；
- 有相对时间且有消息时间戳，则归一化为绝对时间；
- 只能得到月份或年份时，可以使用 YYYY-MM 或 YYYY；
- 日期范围可以使用 YYYY-MM-DD/YYYY-MM-DD；
- 没有时间信息则填 "";
- 不要使用当前系统时间作为记忆时间，除非它就是消息时间戳。

示例：
- 消息时间为 2023-05-08，原文说 yesterday，则 time = "2023-05-07"；
- 消息时间为 2023-07-12，原文说 next month，则 time = "2023-08"；
- 消息时间为 2023-06-09，原文说 last week，则 time = "2023-05-29/2023-06-04"。

6. dimension.location

表示物理地点、线上平台、组织场景、家庭空间、工作场所、系统环境或活动场所。

规则：
- 只有原文明确提到或强烈暗示地点、平台或场景时才填写；
- 不要把普通主题强行写入 location；
- 没有地点信息则填 ""。

7. dimension.reason

表示原因、动因、触发因素或背景条件。

规则：
- 只有原文明确说明或上下文强烈支持时才填写；
- 可以写“某人是在回答某人的问题”这类低风险上下文原因；
- 不要推测隐藏心理动机；
- 没有原因信息则填 ""。

8. dimension.purpose

表示目标、意图或期望结果。

规则：
- 只有原文明确说明或上下文强烈支持时才填写；
- 可以写分享经历、回答问题、解释计划、表达价值观、鼓励对方等低风险交际目的；
- 不要推测未说明的深层目的；
- 没有目的信息则填 ""。

9. dimension.keywords

表示关键检索词或关键短语。

规则：
- keywords 必须是简短词语或名词短语；
- 可以包含人物、地点、活动、对象、事件、关系、照片、视频、目标、偏好、价值观等；
- 不要放入完整句子；
- 不要重复关键词；
- 不要提取无检索价值的普通词；
- 没有明确关键词则填 []。

========================
三、抽取目标
========================

应该抽取：
1. 稳定信息：身份、关系、背景、拥有物、当前状态等。
   例：Alex has a close relationship with his mentor.

2. 事件经历：行为、见面、活动、购买、旅行、分享、计划等。
   例：Nora volunteered at an animal shelter on 2023-07-11.

3. 长期画像：偏好、习惯、兴趣、价值观、目标、能力特征、长期动力来源等。
   例：Ethan regards his family and friends as important sources of support.

4. 重要细节：照片、图片、视频、宠物、物品、地点、陪伴对象、支持系统等，只要承载了具体经历或人物关系，也应抽取。
   例：Mia shared a photo from a meetup last week.

不要抽取：
1. 纯问候、纯感谢、纯确认。
   例：Hi, how are you? / Thanks! / OK, sure.

2. 没有长期价值的孤立情绪或泛泛评价。
   例：That’s nice. / I’m happy today. / It was great.

注意：
不要因为一句话以 Thanks、Haha、OK 等开头就跳过整条消息；只要后续包含有价值信息，就应抽取。

========================
四、抽取规则
========================

1. 按对话顺序处理所有消息。
2. 多人对话中，只要某个说话人的消息包含有价值信息，就应抽取。
3. 每条 memory 只表达一个核心事实、事件、计划或画像特征。
4. 一句话包含多个独立信息时，应拆成多条 memory。
5. 多条信息高度重叠时，可以合并为一条 memory。
6. content 必须自包含，不能依赖原始上下文。
7. 时间要尽量归一化，content 和 dimension.time 应保持一致。
8. 不要过度概括：单次事件不能写成长期画像，临时情绪不能写成稳定偏好。
9. 不要幻觉补全字段：原文无依据的 time、location、reason、purpose 填 ""。
10. 最终只输出合法 JSON，不要输出注释、Markdown、解释或 JSON 以外的内容。

{{OverlappingContextRules}}

========================
六、简短示例
========================

示例 1：

[2023-06-09T19:55:05, Fri] 6.Alex: Thanks! My parents and coach have always pushed me to keep going. I also sent you a photo from our meetup last week.

正确抽取方向：
- “Alex's parents and coach are long-term sources of support and motivation” 应抽取为 profile；
- “Alex sent a photo from a meetup last week” 应抽取为 episodic；
- last week 应根据消息时间归一化为具体日期范围；
- 不要因为消息以 Thanks 开头就跳过整条消息。

示例 2：

[2023-08-02T10:30:00, Wed] 6.Lena: Haha yeah, I bought that blue notebook in Kyoto last month and still use it for project ideas.

正确抽取方向：
- “Lena bought a blue notebook in Kyoto in 2023-07” 应抽取为 episodic；
- “Lena uses the blue notebook for project ideas” 如果表达当前持续用途，可抽取为 fact；
- 不要因为消息以 Haha 开头就跳过后续信息。

以下是你现在需要处理的真实输入：

{{conversation}}
"""


OverlappingContextRules = """
========================
五、重叠上下文规则
========================

输入中前 5 条消息，即编号 1、2、3、4、5，是与上一段对话片段重叠的上下文，主要用于理解第 6 条及之后的消息。

默认不要从前 5 条消息中抽取记忆，除非第 6 条及之后的消息必须依赖它们才能形成完整、自包含的记忆。
""" 

LOCOMO_QUERY_ANALYSIS_PROMPT = """
你是记忆查询解析器。将自然语言问题转换为结构化检索 query。只输出合法 JSON。

== 输出格式 ==

{
  "query_anchor": "",
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
示例 true: "Can you remind me which airline you suggested last time for budget flights?"
示例 false: "How many countries have I visited this year?"

2. dimension.target_memory_type
优先检索的记忆类型，可多选：
- fact: 稳定事实、身份、关系、状态、拥有物
- episodic: 具体事件、经历、行为、购买、旅行、进展
- profile: 偏好、习惯、兴趣、目标、风格
不确定时填 []。

3. dimension.keywords
提取关键实体、人物、物品、工具、地点、活动、主题等短语。无则填 []。

4. dimension.time
仅当存在明确时间约束时填写。格式："on/before/after/around <时间>" 或 "between <起> and <止>"。
- 若有 question_date，归一化相对时间（today/yesterday/this week/last month 等 -> 具体日期）
- 问题在问时间本身时不填此字段，改设 answer_dim = "time"
- 频率词（daily/weekly/often）不算时间约束
- 无明确约束时填 ""

5. dimension.location
仅当存在明确地点/平台/场景约束时填写。问地点本身时不填，改设 answer_dim = "location"。无则填 ""。

6. answer_dim
答案对应的记忆字段：
- "content": 事实/事件/画像内容
- "time": 时间/日期/频率
- "location": 地点/平台/场景
- "reason": 原因
- "purpose": 目的/用途
- "keywords": 人物/物品/名称/工具等关键对象
- "": 需计算/比较/排序/推理/推荐/汇总

== 输入 ==

Question:
{question}
"""
