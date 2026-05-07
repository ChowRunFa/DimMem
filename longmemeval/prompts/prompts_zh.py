LONGMEMEVAL_STRUCTURED_MEMORY_EXTRACTION_PROMPT = """
你是一个结构化记忆抽取器。

你的任务是：根据输入的用户—助手对话记录，抽取有价值的结构化记忆，并严格按照指定 JSON 格式输出。

你只能输出合法 JSON，不要输出任何解释、分析、Markdown 或额外文本。

========================
一、输出格式
========================

输出必须是如下 JSON 结构：

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
    },
    {
      "source_id": 2,
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
二、抽取目标
========================
应该抽取：
1. 事实信息；
2. 具体经历、事件、行为、计划；
3. 长期偏好、习惯、兴趣、价值观、目标；
4. 对未来理解用户需求有帮助的信息。
5. 如果不是明确的无价值信息，就需要抽取对应记忆
不要抽取：
1. 问候语、感谢、简单确认、无意义闲聊；
2. 临时性格式要求或一次性操作指令；

========================
三、memory_type 定义
========================

dimension.memory_type 只能是以下三类之一：fact、episodic、profile。不得输出其他类型。

1. fact：事实记忆

用于记录相对稳定、明确、可复述的事实信息。

包括：
- 身份、背景、关系；
- 当前状态；
- 使用的工具、模型、数据集；
- 项目配置；
- 已确定的选择；
- 稳定客观属性。

判断标准：
如果信息主要回答“是什么 / 有什么 / 使用什么 / 关系是什么 / 当前状态是什么”，标为 fact。

示例：
- The user uses LLaMA2-7B as the base model.
- Caroline has two children.
- Melanie is Caroline's close friend.

2. episodic：情景记忆

用于记录具体发生过、正在发生或计划发生的事件、经历、行为、阶段性动作。

包括：
- 一次具体事件；
- 一段经历；
- 一次行动；
- 阶段性进展；
- 未来计划；
- 有时间、地点或场景属性的具体事实。

判断标准：
如果信息主要回答“发生了什么 / 做了什么 / 经历了什么 / 计划做什么 / 什么时候发生 / 在哪里发生”，标为 episodic。

示例：
- Marry attended a meeting about project management on May 7, 2023.
- Evan took his family on a road trip to Jasper.
- The user plans to train a local LLaMA2-7B model using Urdu data.

3. profile：用户画像

用于记录长期稳定或反复出现的用户特征。

包括：
- 偏好；
- 习惯；
- 兴趣；
- 价值观；
- 长期目标；
- 能力特征；
- 写作风格偏好；
- 交互偏好；
- 稳定行为模式。

判断标准：
如果信息主要回答“长期是什么样 / 喜欢什么 / 通常做什么 / 相信什么 / 重视什么 / 偏好什么风格”，标为 profile。

示例：
- The user prefers concise Java code with a single main function.
- Maria practices yoga every Saturday morning.
- Caroline believes every child deserves love, acceptance, and a safe home.

三类边界：

fact = 是什么 / 有什么 / 使用什么 / 关系是什么
episodic = 发生了什么 / 做了什么 / 经历了什么 / 计划做什么
profile = 长期是什么样 / 喜欢什么 / 通常做什么 / 相信什么

如果无法归入 fact、episodic 或 profile，不要抽取。

========================
四、content 字段定义
========================

content 是该条记忆的主文本表示。

要求：

1. 必须是完整、自包含、可检索、可直接用于回答问题的一句话；
2. 必须明确说明这条记忆关于谁；
3. 必须明确说明核心事实、事件或画像内容；
4. 如果原文包含时间、地点、原因、目的，应尽量写入 content；
5. 消除模糊代词，例如“他”“她”“它”“那里”“这个”“那个”；
6. 相对时间应根据消息时间戳归一化，例如 yesterday → 具体日期；
7. 不要加入原文没有支持的信息；
8. 不要把单次事件写成长期画像；
9. 不要把临时情绪写成稳定偏好。

不同 memory_type 的 content 写法：

1. fact：写成稳定事实命题。

格式：[主体] 是 / 有 / 使用 / 属于 / 与 [对象] 存在 [关系]。

示例：The user uses LLaMA2-7B as the base model and trains it with LoRA.

2. episodic：写成具体事件记录。

格式：[主体] 在 [时间] 于 [地点] 做了 / 经历了 / 计划做 [事件]，并保留结果、感受、原因或目的。

示例：Marry attended a meeting about project management on May 7, 2023, and found the experience powerful.

3. profile：写成长期画像描述。

格式：[主体] 喜欢 / 偏好 / 通常 / 经常 / 相信 / 重视 / 长期关注 [对象、行为或价值]。

示例：The user prefers concise Java code, ideally implemented with a single main function.

========================
五、dimension 字段定义
========================

dimension 用于记录结构化检索维度。

除 memory_type 外，如果字段没有明确依据，使用空字符串 ""。
keywords 如果没有明确关键词，使用空数组 []。

1. time

表示记忆成立、发生、计划发生或重复出现的时间。

- fact：事实成立时间、状态开始时间、配置时间；
- episodic：事件发生时间、经历时间、计划时间；
- profile：习惯频率、长期持续时间、画像特征适用时间。

规则：
- 有绝对日期则使用绝对日期；
- 有相对时间且有消息时间戳，则归一化为绝对时间；
- 只能得到模糊时间时，可保留模糊表达；
- 没有时间信息则填 ""；
- 不要使用当前系统时间作为记忆时间，除非它就是消息时间戳。

2. location

表示物理地点、线上平台、组织场景、家庭空间、工作场所、系统环境或活动场所。

- fact：事实适用的场景、系统环境、组织环境或项目环境；
- episodic：事件发生或计划发生的地点、场所或平台；
- profile：偏好、习惯或画像特征通常适用的场景。

规则：
- 只有原文明确提到或强烈暗示地点、平台或场景时才填写；
- 不要把普通主题强行写入 location；
- 没有地点信息则填 ""。

3. reason

表示原因、动因、触发因素或背景条件。

- fact：事实、状态或配置形成的原因；
- episodic：事件发生的原因、触发因素或背景原因；
- profile：偏好、习惯、价值观或长期行为模式形成的原因。

规则：
- 只有原文明确说明或强烈暗示原因时才填写；
- 不要推测隐藏动机；
- 不要和 purpose 混淆；
- 没有原因信息则填 ""。

4. purpose

表示目标、意图或期望结果。

- fact：事实、配置、工具选择或状态服务的目标；
- episodic：某次事件、行为或计划的目标；
- profile：长期偏好、习惯或行为模式服务的稳定目标。

规则：
- 只有原文明确说明或强烈暗示目标时才填写；
- 不要推测未说明的目的；
- 没有目的信息则填 ""。

5. keywords

表示该条记忆中的关键检索词或关键短语，用于检索、去重和 query-memory 对齐。

不同 memory_type 下的 keywords 重点：

- fact：主体、对象、工具、模型、数据集、项目、关系对象、稳定属性；
- episodic：事件参与者、地点、活动、关键对象、事件结果、具体行为；
- profile：画像主体、偏好对象、习惯活动、兴趣领域、价值观对象、长期目标对象。

规则：
- keywords 必须是简短词语或名词短语；
- 可以包含重要实体，也可以包含有检索价值的主题词、行为词或对象词；
- 不要把完整句子放入 keywords；
- 不要重复关键词；
- 不要提取无检索价值的普通词；
- 没有明确关键词则填 []。

========================
六、抽取规则
========================

1. 按对话顺序处理消息。
2. 主要从用户消息中抽取记忆。
3. 每条 memory 应尽量原子化。如果一句话包含多个独立记忆，应拆成多条 memory。
4. content 必须自包含，不能依赖原始对话上下文。
5. 不要过度概括。单次事件不能变成长期画像，临时情绪不能变成稳定偏好。
6. 不要幻觉补全字段。原文没有依据的 time、location、reason、purpose 必须留空。
7. 时间要尽量归一化。若消息时间为 2023-05-08，原文说 yesterday，则 time 应为 2023-05-07。
8. 只抽取有长期价值的信息。临时格式要求、当前任务步骤、简单确认、无意义闲聊一般不抽取。
9. 输出必须是合法 JSON，不要输出注释、Markdown、解释或 JSON 以外的任何文本。

========================
七、输入与重叠上下文规则
========================

你将收到一个“当前对话片段”作为输入。

重要规则：
1. 输入中前 3 条消息（即标号 1、2、3）是与上一段提取记忆时重叠的上下文。
2. 这 3 条消息不能作为新的记忆抽取来源。
3. 你只能把它们当作辅助上下文，用来理解后续消息。
4. 真正允许抽取的内容，从第 4 条消息开始。
5. 如果第 4 条及之后的内容重复了前 3 条中的信息，只有在后文提供了更完整、更明确、可独立成记忆的新信息时，才允许抽取。

示意格式如下：
[2023-07-12T08:10:00, Wed] 1.User: 我最近在调整早晨作息，想把起床后的前一小时安排得更稳定一些。之前我总会一边刷短视频一边拖延早餐，结果每天出门都很赶。你能推荐一些帮助我建立晨间习惯的方法吗？
[2023-07-12T08:10:00.500000, Wed] 2.User: 设定固定的晨间流程听起来不错。我还想把冥想和简单拉伸加进去，这样也许能让我白天更专注。你能帮我设计一个大概二十分钟的晨间流程吗？
[2023-07-12T08:10:01, Wed] 3.User: 我也在考虑重新开始记手账，把每天最重要的三件事写下来。你能给我一些适合早晨快速记录的模板吗？
[2023-07-12T08:10:01.500000, Wed] 4.User: 其实我发现，暂停刷短视频以后，我早上准备出门的速度快了很多。以前我几乎每天早上都会花将近一个小时刷视频，现在我想继续控制这个习惯，避免重新反弹。
[2023-07-12T08:10:02, Wed] 5.User: 为了替代刷视频的时间，我最近开始在早餐前读书二十分钟，感觉注意力比以前集中一些。我想把这个习惯稳定下来。
[2023-07-12T08:10:02.500000, Wed] 6.User: 我还准备周末去上一节陶艺体验课，如果喜欢的话，可能会把陶艺发展成新的长期爱好。
[2023-07-12T18:45:00, Wed] 7.User: 我在安排接下来几周的运动计划。你还记得我下次和同事一起打羽毛球是什么时候吗？
[2023-07-12T18:45:00.500000, Wed] 8.User: 我记得好像是下周末。最近我也在增加跑步训练，希望打球的时候体能更稳定一些。

对于上面这种输入：
- 1、2、3 只能帮助你理解上下文，不能从 1、2、3 直接抽取新记忆，只从 4 及之后的消息中抽取。

以下是你现在需要处理的真实输入：

{conversation}
"""

LONGMEMEVAL_QUERY_ANALYSIS_PROMPT = """
你是一个结构化记忆查询解析器。

你的任务是：将自然语言问题转换为用于记忆检索的结构化 query。

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
字段定义
========================

1. parse_mode

parse_mode 用于决定 query 的检索解析方式，只能是 structured、hybrid、raw 三类之一。

structured：
问题结构清晰，答案可以从一条或少量 memory 中直接抽取，并且能够明确对应memory.dimension 的字段。
适合普通事实、时间、地点、原因、目的、人物/物品/名称、具体事件或画像内容问答。
如果答案能直接对应 content、time、location、reason、purpose、keywords，优先选择 structured。

hybrid：
问题有明确检索目标，但答案需要多条 memory 聚合、比较、排序、计算或推理后得到。
适合 total、in total、combined、average、difference、percentage、more/less than、before/after、earliest/latest、order、how many times、how much more 等问题。
如果答案不能直接对应单个 memory 字段，但检索目标明确，选择 hybrid。

raw：
问题不适合可靠结构化解析，主要依赖 query_anchor 和 keywords 做 BM25 + embedding 宽召回。
适合推荐、建议、开放规划、模糊回忆、强依赖 previous/last time/that/you recommended 的问题。
如果强行填结构化字段会引入错误约束，选择 raw。

2. query_anchor

从原始问题改写得到的语义检索 query，用于做语义匹配。

query_anchor 不是关键词列表，而是一个更适合检索的自然语言查询表达。

要求：
- 保留原问题的核心意图；
- 明确主体、动作、对象和关键上下文；
- 将第一人称表达归一化，例如将 “I / me / my” 改写为 “the user”；
- 保留重要的时间、地点、比较、数量、顺序信息；

How many weeks ago did I start using the cashback app 'Ibotta'? -> How many weeks ago did the user start using the cashback app 'Ibotta'?

3. dimension.target_memory_type

表示优先检索的 memory 类型。

允许值：

- fact：稳定事实、身份、背景、关系、当前状态、工具、模型、数据集、配置、拥有物或稳定属性。
- episodic：具体事件、经历、行为、购买、旅行、计划、阶段进展、时间线相关记忆。
- profile：偏好、习惯、兴趣、价值观、长期目标、能力特征、风格偏好、稳定行为模式。

可以选择一个或多个。如果问题较模糊或主要走 raw 检索，可以填 []。

4. dimension.keywords

表示问题中的关键检索词或关键短语，用于检索、去重和 query-memory 对齐。

提取重点：
- fact 类问题：主体、对象、工具、模型、数据集、项目、关系对象、稳定属性；
- episodic 类问题：事件参与者、地点、活动、关键对象、事件结果、具体行为；
- profile 类问题：画像主体、偏好对象、习惯活动、兴趣领域、价值观对象、长期目标对象。

规则：
- keywords 必须是简短词语或名词短语；
- 可以包含重要实体，也可以包含有检索价值的主题词、行为词或对象词；
- 不要把完整句子放入 keywords；
- 不要重复关键词；
- 不要提取没有检索价值的普通词；
- 没有明确关键词时填 []。

5. dimension.time

time 表示问题中的时间约束，用于匹配 memory.dimension.time。

time 只能在问题中出现明确时间约束，或出现可基于 question_date 可靠归一化的相对时间约束时填写。

time 必须使用以下格式之一：

- "on <具体时间>"
- "before <具体时间>"
- "after <具体时间>"
- "around <具体时间>"
- "between <开始时间> and <结束时间>"
- ""

其中 <具体时间> 应尽量写成可比较的时间表达，例如：
- "2023-05-08"
- "2023-05"
- "2023"
- "May 8, 2023"

对于可归一化的相对时间表达，必须根据 question_date 转换为具体时间范围：

- this year：转换为 "between YYYY-01-01 and YYYY-12-31"
- this month：转换为 "between YYYY-MM-01 and YYYY-MM-最后一天"
- this week：转换为该自然周范围，格式为 "between YYYY-MM-DD and YYYY-MM-DD"
- today：转换为 "on YYYY-MM-DD"
- yesterday：转换为 "on YYYY-MM-DD"
- last year：转换为 "between 上一年-01-01 and 上一年-12-31"
- last month：转换为 "between 上个月第一天 and 上个月最后一天"
- last week：转换为上一自然周范围
- earlier this year：转换为 "between YYYY-01-01 and question_date当天"
- earlier this month：转换为 "between YYYY-MM-01 and question_date当天"

规则：
- 如果问题中出现明确日期、月份或年份，使用 "on <具体时间>"；
- 如果问题中出现 before / after / around / between，并且参照对象是具体时间点、具体日期、具体月份或具体年份，使用对应格式；
- 如果问题中出现 this year / this month / last year / last month / today / yesterday 等相对时间，并且 question_date 可用，必须归一化为具体时间；
- 如果问题是在询问时间，time 不作为约束填写，而应设置 answer_dim = "time"；
- 不要把频率写入 time，例如 every Saturday、usually、often、twice a week、daily、weekly；
- 不要把 latest、earliest、first、previous 单独写入 time，这些属于排序、比较或阶段语义，通常应选择 hybrid；
- 不要把无法归一化为具体时间点或具体时间范围的事件相对时间写入 time；
- 没有明确具体时间约束时填 ""。

6. dimension.location

表示问题中的地点、平台、场景或环境约束，用于匹配 memory.dimension.location。

规则：
- 如果地点是已知检索条件，填入 location；
- 如果问题是在询问地点，location 不作为约束填写，而应设置 answer_dim = "location"；
- 不要把普通主题强行写成 location；
- 没有地点约束时填 ""。

7. answer_dim

表示答案直接对应的 memory 字段。

只有当答案明确可以从 memory.content 或 memory.dimension 中的某个字段直接抽取时才填写。

允许值只能是：

content
time
location
reason
purpose
keywords

如果答案不能直接对应这些字段，必须填 ""。

规则：
- 问普通事实、事件内容、画像内容：answer_dim = "content"；
- 问时间、日期、星期、频率、通常时间：answer_dim = "time"；
- 问地点、平台、场景：answer_dim = "location"；
- 问原因：answer_dim = "reason"；
- 问目的、用途、目标：answer_dim = "purpose"；
- 问人物、物品、组织、名称、对象、平台、作品等关键短语：answer_dim = "keywords"；
- 如果问题需要统计、求和、计算金额、计算时长、比较、排序、判断 yes/no、推荐、汇总或复杂推理，answer_dim 必须填 ""。

========================
输出规则
========================

1. 最终只输出合法 JSON。

2. 所有字段都必须出现。

3. parse_mode 必须是 structured、hybrid 或 raw 之一。

4. keywords 必须尽量提取，不能因为 parse_mode 为 raw 就留空。

5. query_anchor 必须是对原问题的语义重写，不是关键词列表。

6. query_anchor 中应将第一人称表达归一化，例如将 “I / me / my” 改写为 “the user”。

7. answer_dim 只有在答案明确对应 memory.content 或 memory.dimension 中的字段时才填写。

8. answer_dim 只能是 content、time、location、reason、purpose、keywords 或空字符串 ""。

9. 如果答案需要统计、计算、比较、排序、判断、推荐、汇总或复杂推理，answer_dim 必须填 ""。

10. dimension.time 只能使用 "on <具体时间>"、"before <具体时间>"、"after <具体时间>"、"around <具体时间>"、"between <开始时间> and <结束时间>" 或 ""。

11. 如果字段没有明确内容：
- target_memory_type 使用 []；
- keywords 尽量不要为空，确实没有可用关键词时使用 []；
- time 使用 "";
- location 使用 "";
- answer_dim 使用 ""。
"""


LONGMEMEVAL_OFFLINE_UPDATE_PROMPT = """

"""




__all__ = [
    "LONGMEMEVAL_OFFLINE_UPDATE_PROMPT",
    "LONGMEMEVAL_QUERY_ANALYSIS_PROMPT",
    "LONGMEMEVAL_STRUCTURED_MEMORY_EXTRACTION_PROMPT",
]
