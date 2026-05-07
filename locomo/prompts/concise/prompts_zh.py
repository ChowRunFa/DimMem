from cgi import MiniFieldStorage
LOCOMO_STRUCTURED_MEMORY_EXTRACTION_PROMPT = """
你是一个结构化记忆抽取器。

任务：从输入的多人对话记录中，尽可能完整地抽取具有长期检索价值的结构化记忆，并严格输出合法 JSON。

核心原则：
- 按消息顺序逐条处理；只要消息包含有意义信息，就应抽取。
- 宁可细粒度多抽，也不要遗漏人物、关系、时间、地点、事件、物品、照片、计划、偏好、原因、目的等有价值细节。
- 只跳过明确无信息量的内容，例如纯寒暄、纯感谢、纯确认或无上下文价值的泛泛评价。
- 不要因为消息以 Thanks、Haha、OK 等开头就跳过整条消息；只要后续内容有信息，就必须抽取。

========================
输出格式
========================

只返回合法 JSON，不要输出解释、分析、Markdown 或额外文本。

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

如果没有值得抽取的记忆，输出：

{
  "memories": []
}

========================
字段说明
========================

1. source_id

该条 memory 主要来源于哪一条消息编号。

2. source_speaker

source_id 对应消息的说话人。

3. content

记忆的核心文本，必须自包含、清晰、可独立理解和检索，并保留原文支持的关键细节。

规则：
- 消解代词和模糊指代，明确人物、对象、地点或事件。
- 保留重要细节，例如时间、地点、人物、关系、物品、活动、照片、视频、计划、原因、目的等。
- 根据消息时间戳归一化相对时间，例如 yesterday → 具体日期，last week → 日期范围，next month → 具体月份。
- 不添加原文不支持的信息，不把单次事件过度概括为长期画像。

4. dimension.memory_type

只能是 fact、episodic、profile 三类之一。

- fact：稳定事实、身份、关系、背景、当前状态、拥有物、工具、模型、数据集或配置。
- episodic：具体事件、经历、行为、见面、购买、旅行、分享、计划、阶段进展，或与具体事件相关的照片、图片、视频等媒介。
- profile：长期偏好、习惯、兴趣、价值观、目标、能力特征、交互偏好、行为模式或稳定支持来源。

5. dimension.time

记忆成立、发生、计划发生或重复出现的时间。

规则：
- 有绝对日期则使用绝对日期。
- 有相对时间则根据消息时间戳归一化。
- 可使用 YYYY-MM-DD、YYYY-MM、YYYY 或 YYYY-MM-DD/YYYY-MM-DD 表示日期范围。
- 没有时间依据则填 ""。
- content 中的时间描述必须与 dimension.time 一致。

6. dimension.location

原文明确提到或强烈暗示的物理地点、线上平台、组织场景、家庭空间、工作场所、系统环境或活动场所。

7. dimension.reason

原文明确说明或上下文强烈支持的原因、动因、触发因素或背景条件。

8. dimension.purpose

原文明确说明或上下文强烈支持的目标、意图或期望结果。

9. dimension.keywords

用于检索的简短关键词或名词短语，例如人物、地点、活动、对象、事件、关系、照片、目标、偏好或价值观。

========================
抽取规则
========================

1. 按消息顺序处理所有消息。
2. 多人对话中，任何说话人的消息只要包含有价值信息，都应抽取。
3. 每条 memory 只表达一个核心事实、事件、计划或画像特征。
4. 一句话包含多个独立信息时，拆成多条 memory。
5. 多条信息高度重叠时可以合并，但不能丢失关键细节。
6. content 必须自包含，不能依赖原始对话上下文。
7. 尽量归一化时间，并保持 content 与 dimension.time 一致。
8. 不要幻觉补全字段；无依据的 time、location、reason、purpose 填 ""。
9. 最终只输出合法 JSON。

========================
应抽取的内容
========================

应抽取：
- 身份、关系、背景、拥有物、当前状态。
- 见面、活动、购买、旅行、分享、计划、阶段进展。
- 偏好、习惯、兴趣、价值观、长期目标、长期支持来源。
- 照片、图片、视频、宠物、物品、地点、陪伴对象、支持系统等有检索价值的细节。

不要抽取：
- 纯问候：Hi / Hello / How are you?
- 纯感谢：Thanks / Thank you
- 纯确认：OK / Sure / Got it
- 无上下文价值的泛泛评价：That’s nice / Sounds good / Great

{{OverlappingContextRules}}

========================
示例
========================

输入：
[2023-06-09T19:55:05, Fri] 6.Alex: Thanks! My parents and coach have always pushed me to keep going. I also sent you a photo from our meetup last week.

正确抽取方向：
- Alex's parents and coach are long-term sources of support and motivation. → profile
- Alex sent a photo from a meetup during 2023-05-29/2023-06-04. → episodic
- 不要因为消息以 Thanks 开头就跳过。
- last week 必须根据消息时间戳归一化。

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

