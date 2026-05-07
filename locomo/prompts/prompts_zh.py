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

LONGMEMEVAL_QUERY_ANALYSIS_PROMPT = """
你是一个结构化记忆查询解析器。

任务：将自然语言问题转换为用于记忆检索的结构化 query。

该 query 将用于匹配如下 memory 结构：

{
  "memory": {
    "content": "",
    "dimension": {
      "memory_type": "fact | episodic | profile",
      "time": "",
      "location": "",
      "reason": "",
      "purpose": "",
      "keywords": []
    }
  }
}

只能输出合法 JSON，不要输出解释、Markdown 或额外文本。

========================
输出格式
========================

{
  "parse_mode": "",
  "query_anchor": "",
  "dimension": {
    "target_memory_type": [],
    "keywords": [],
    "time": "",
    "location": ""
  },
  "answer_dim": ""
}

========================
字段说明
========================

1. parse_mode

只能是 structured、hybrid、raw 之一。

structured：
问题目标清晰，答案可从一条或少量 memory 中直接抽取，并能明确对应 content、time、location、reason、purpose 或 keywords。
适合事实、事件、时间、地点、原因、目的、人物、物品、名称、画像内容等问答。

hybrid：
问题有明确检索目标，但答案需要多条 memory 聚合、统计、比较、排序、计算或推理后得到。
适合 how many times、total、average、difference、percentage、more/less than、before/after、earliest/latest、first/last/order、based on 等问题。

raw：
问题不适合可靠结构化解析，主要依赖 query_anchor 和 keywords 做 BM25 + embedding 宽召回。
适合推荐、建议、开放规划、模糊回忆、强上下文依赖或无法明确结构化约束的问题。
如果强行填写结构化字段会引入错误约束，选择 raw。

2. query_anchor

从原始问题改写得到的自然语言检索 query，不是关键词列表。

要求：
- 保留原问题核心意图；
- 保留原问题中的人物名称、对象、事件和关键上下文；
- 不要改变具体主体名称；
- 可以适度补全省略信息，使 query 更适合检索；
- 保留重要的时间、地点、比较、数量、顺序信息。

示例：

Why did Sophie visit the design studio?->What was Sophie's purpose for visiting the design studio?

What recipe did Nathan try after the cooking workshop?->What recipe did Nathan try after attending the cooking workshop?

Which museum would Clara likely enjoy visiting based on her interest in modern sculpture?->Which museum would Clara likely enjoy visiting based on Clara's interest in modern sculpture?

3. dimension.target_memory_type

表示优先检索的 memory 类型，可多选；如果不明确或主要使用 raw 检索，填 []。

允许值：
- fact：稳定事实、身份、背景、关系、当前状态、工具、模型、数据集、配置、拥有物、稳定属性。
- episodic：具体事件、经历、行为、购买、旅行、计划、阶段进展、时间线相关记忆。
- profile：偏好、习惯、兴趣、价值观、长期目标、能力特征、风格偏好、稳定行为模式。

4. dimension.keywords

表示问题中的关键检索词或短语，用于召回、去重和 query-memory 对齐。

提取重点：
- fact：人物、对象、工具、模型、数据集、项目、关系对象、稳定属性；
- episodic：事件、参与者、地点、活动、关键对象、结果、具体行为；
- profile：偏好对象、习惯活动、兴趣领域、价值观对象、长期目标对象。

规则：
- 使用简短词语或名词短语；
- 保留重要人物名；
- 可包含重要实体、主题词、行为词或对象词；
- 不要放完整句子；
- 不要重复；
- 不要提取无检索价值的普通词；
- 没有明确关键词时填 []；
- 即使 parse_mode 为 raw，也应尽量提取 keywords。

5. dimension.time

表示问题中的时间约束，用于匹配 memory.dimension.time。

只有当问题本身包含明确时间约束时才填写。

允许格式只能是：
- "on <具体时间>"
- "before <具体时间>"
- "after <具体时间>"
- "around <具体时间>"
- "between <开始时间> and <结束时间>"
- ""

时间表达应尽量可比较，例如：
- "2023-05-08"
- "2023-05"
- "2023"
- "May 8, 2023"
- "March"

规则：
- 如果问题中出现明确日期、月份、年份或可比较时间范围，填入 time；
- 如果问题是在询问时间，time 不作为约束填写，应设置 answer_dim = "time"；
- 如果没有 question_date，不要归一化 today、yesterday、this month、last week、currently、now、recently 等相对时间；
- currently、now、recently 通常表示当前状态或近况，不写入 time；
- 不要把 every Saturday、usually、often、twice a week、daily、weekly 等频率写入 time；
- 不要把 latest、earliest、first、previous 单独写入 time，这类通常属于 hybrid；
- 没有明确时间约束时填 ""。

示例：

When did Olivia start her online language course?
- time = ""
- answer_dim = "time"

Which month did Marcus begin training for the marathon?
- time = ""
- answer_dim = "time"

What podcast is Hannah currently listening to?
- time = ""
- answer_dim = "content"

What did Leo buy in April?
- time = "on April"
- answer_dim = "content"

What project did Emma work on before 2022?
- time = "before 2022"
- answer_dim = "content"

6. dimension.location

表示问题中的地点、平台、场景或环境约束，用于匹配 memory.dimension.location。

规则：
- 如果地点、平台或场景是检索条件，填入 location；
- 如果问题是在询问地点，location 不作为约束填写，应设置 answer_dim = "location"；
- 不要把普通主题强行写成 location；
- 没有地点约束时填 ""。

示例：

Where is Aaron's photography studio?
- location = ""
- answer_dim = "location"

What restaurant did Mia recommend in Chicago?
- location = "Chicago"
- answer_dim = "keywords"

Which bookstore would Rachel enjoy visiting in London City based on her reading interests?
- location = "London"
- answer_dim = ""

7. answer_dim

表示答案直接对应的 memory 字段。

允许值只能是：
- "content"
- "time"
- "location"
- "reason"
- "purpose"
- "keywords"
- ""

填写规则：
- 问普通事实、事件内容、画像内容：content
- 问时间、日期、月份、星期、频率、通常时间：time
- 问地点、平台、场景：location
- 问原因：reason
- 问目的、用途、目标：purpose
- 问人物、物品、组织、名称、对象、平台、作品等关键短语：keywords
- 如果问题需要统计、求和、计算、比较、排序、yes/no 判断、推荐、推测、汇总或复杂推理，填 ""。


以下是要解析的真正问题：
Question:
{{question}}
"""