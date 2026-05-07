LOCOMO_STRUCTURED_MEMORY_EXTRACTION_PROMPT = f"""
You are a structured memory extractor.

Your task is to extract structured memories with long-term value from the input multi-person conversation records, and strictly output valid JSON.

You may only output JSON. Do not output explanations, analysis, Markdown, or any extra text.

========================
1. Output Format
========================

Return only the following JSON object:

{{
  "memories": [
    {{
      "source_id": 1,
      "source_speaker": "",
      "content": "",
      "dimension": {{
        "memory_type": "",
        "time": "",
        "location": "",
        "reason": "",
        "purpose": "",
        "keywords": []
      }}
    }}
  ]
}}

========================
2. Field Descriptions
========================

1. source_id

Indicates which message number in the input conversation this memory mainly comes from.

2. source_speaker

Indicates the speaker of the message corresponding to source_id.

3. content

content is the core textual representation of the memory. It must be complete, self-contained, independently understandable, and retrievable.

The core requirements are:

- Coreference resolution: clearly state who the subject is and what the core information is. Do not use ambiguous references such as "he", "she", "there", or "this". Restore specific people, objects, locations, or events based on context.

- Key information preservation: retain key elements from the original text as much as possible, including time, location, reason, purpose, object, interpersonal relationship, media evidence, etc. Do not add information that is not supported by the original text.

- Time normalization: when content involves time, normalize it based on the message timestamp.
  For example: yesterday → a specific date; last week → a date range; next month → a specific month; three years ago → a specific or approximate year.

4. dimension.memory_type

memory_type must be one of the following three types: fact, episodic, profile.

fact:
Used to record stable facts, identity, background, relationships, current status, tools, models, datasets, configurations, possessions, or stable attributes.

Judgment criterion:
If the information mainly answers "what it is / what someone has / what is used / what the relationship is / what the current status is", label it as fact.

Examples:
- Jack has two children.
- Mico is Jack's close friend.

episodic:
Used to record specific events, experiences, actions, purchases, trips, meetings, sharing, plans, stage progress, or timeline-related information.

Judgment criterion:
If the information mainly answers "what happened / what someone did / what someone experienced / what someone plans to do / when it happened / where it happened", label it as episodic.

Notes:
- Media such as photos, images, videos, screenshots, and audio should also be extracted as episodic if they carry a specific experience, meeting, activity, trip, family event, friend interaction, or important object;
- Do not ignore information simply because it is about "taking a photo", "sending an image", or "sharing a photo". As long as it points to a retrievable specific experience, it should be retained.

Examples:
- Nora volunteered at an animal shelter on 2023-07-11.
- Ethan joined a river cleanup with his sister on 2023-07-30.
- Maria shared a photo from a family dinner in May 2023.

profile:
Used to record long-term preferences, habits, interests, values, long-term goals, ability traits, interaction preferences, stable behavior patterns, or long-term sources of support.

Judgment criterion:
If the information mainly answers "what someone is like over the long term / what someone likes / what someone usually does / what someone believes / what someone values / what someone relies on / what motivates someone", label it as profile.

Notes:
- Friends, family members, mentors, partners, colleagues, pets, etc., should be extracted as profile or fact if they are described as important support, sources of motivation, long-term companions, or stable relationships;
- If it is only a one-time meeting or interaction, label it as episodic;
- If it expresses a long-term relationship or stable influence, label it as profile or fact.

Examples:
- Maria practices yoga every Saturday morning.
- Mico believes every child deserves love, acceptance, and a safe home.
- Ethan sees his older sister as an important source of support.

Do not extract content that cannot be classified as fact, episodic, or profile.

5. dimension.time

Indicates the time when the memory is valid, happened, is planned to happen, or recurs.

Rules:
- Use an absolute date if one is available;
- If there is relative time and a message timestamp is available, normalize it to absolute time;
- If only a month or year can be obtained, use YYYY-MM or YYYY;
- Date ranges may use YYYY-MM-DD/YYYY-MM-DD;
- If there is no time information, fill in "";
- Do not use the current system time as the memory time unless it is the message timestamp.

Examples:
- If the message time is 2023-05-08 and the original text says yesterday, then time = "2023-05-07";
- If the message time is 2023-07-12 and the original text says next month, then time = "2023-08";
- If the message time is 2023-06-09 and the original text says last week, then time = "2023-05-29/2023-06-04".

6. dimension.location

Indicates the physical location, online platform, organizational context, home space, workplace, system environment, or activity venue.

Rules:
- Fill this field only when the original text explicitly mentions or strongly implies a location, platform, or context;
- Do not force ordinary topics into location;
- If there is no location information, fill in "".

7. dimension.reason

Indicates the reason, motivation, trigger, or background condition.

Rules:
- Fill this field only when the original text explicitly states it or the context strongly supports it;
- Low-risk contextual reasons such as "someone was answering someone's question" may be used;
- Do not infer hidden psychological motives;
- If there is no reason information, fill in "".

8. dimension.purpose

Indicates the goal, intention, or expected result.

Rules:
- Fill this field only when the original text explicitly states it or the context strongly supports it;
- Low-risk communicative purposes such as sharing an experience, answering a question, explaining a plan, expressing values, or encouraging the other person may be used;
- Do not infer unstated deeper purposes;
- If there is no purpose information, fill in "".

9. dimension.keywords

Indicates key retrieval terms or key phrases.

Rules:
- keywords must be short terms or noun phrases;
- They may include people, locations, activities, objects, events, relationships, photos, videos, goals, preferences, values, etc.;
- Do not include complete sentences;
- Do not repeat keywords;
- Do not extract generic words with no retrieval value;
- If there are no clear keywords, fill in [].

========================
3. Extraction Targets
========================

You should extract:

1. Stable information: identity, relationships, background, possessions, current status, etc.
   Example: Alex has a close relationship with his mentor.

2. Event experiences: actions, meetings, activities, purchases, trips, sharing, plans, etc.
   Example: Nora volunteered at an animal shelter on 2023-07-11.

3. Long-term profiles: preferences, habits, interests, values, goals, ability traits, long-term sources of motivation, etc.
   Example: Ethan regards his family and friends as important sources of support.

4. Important details: photos, images, videos, pets, objects, locations, companions, support systems, etc., should also be extracted as long as they carry a specific experience or interpersonal relationship.
   Example: Mia shared a photo from a meetup last week.

Do not extract:

1. Pure greetings, pure thanks, or pure confirmations.
   Example: Hi, how are you? / Thanks! / OK, sure.

2. Isolated emotions or generic evaluations with no long-term value.
   Example: That’s nice. / I’m happy today. / It was great.

Note:
Do not skip an entire message just because it starts with Thanks, Haha, OK, etc. As long as the following content contains valuable information, it should be extracted.

========================
4. Extraction Rules
========================

1. Process all messages in conversation order.
2. In a multi-person conversation, extract from any speaker's message as long as it contains valuable information.
3. Each memory should express only one core fact, event, plan, or profile trait.
4. If one sentence contains multiple independent pieces of information, split them into multiple memories.
5. If multiple pieces of information highly overlap, they may be merged into one memory.
6. content must be self-contained and must not depend on the original context.
7. Normalize time as much as possible, and keep content consistent with dimension.time.
8. Do not overgeneralize: a one-time event must not be written as a long-term profile, and a temporary emotion must not be written as a stable preference.
9. Do not hallucinate field values: fill unsupported time, location, reason, and purpose fields with "".
10. The final output must be valid JSON only. Do not output comments, Markdown, explanations, or anything outside JSON.

{{OverlappingContextRules}}

========================
6. Short Examples
========================

Example 1:

[2023-06-09T19:55:05, Fri] 6.Alex: Thanks! My parents and coach have always pushed me to keep going. I also sent you a photo from our meetup last week.

Correct extraction direction:
- "Alex's parents and coach are long-term sources of support and motivation" should be extracted as profile;
- "Alex sent a photo from a meetup last week" should be extracted as episodic;
- last week should be normalized into a specific date range based on the message time;
- Do not skip the entire message just because it starts with Thanks.

Example 2:

[2023-08-02T10:30:00, Wed] 6.Lena: Haha yeah, I bought that blue notebook in Kyoto last month and still use it for project ideas.

Correct extraction direction:
- "Lena bought a blue notebook in Kyoto in 2023-07" should be extracted as episodic;
- "Lena uses the blue notebook for project ideas" may be extracted as fact if it expresses a current ongoing use;
- Do not skip the following information just because the message starts with Haha.

The following is the real input you need to process now:

{{conversation}}
"""

OverlappingContextRules = """
========================
5. Overlapping Context Rule
========================

The first 5 messages in the input, namely messages numbered 1, 2, 3, 4, and 5, are overlapping context from the previous conversation segment. They are mainly used to understand message 6 and later messages.

By default, do not extract memories from the first 5 messages, unless messages 6 and later must rely on them to form a complete and self-contained memory.
"""

LONGMEMEVAL_QUERY_ANALYSIS_PROMPT = """
You are a structured memory query parser.

Task: Convert a natural language question into a structured query for memory retrieval.

This query will be used to match the following memory structure:

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

Only output valid JSON. Do not output explanations, Markdown, or any extra text.

========================
Output Format
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
Field Descriptions
========================

1. parse_mode

Must be one of: structured, hybrid, raw.

structured:
The question has a clear target. The answer can be directly extracted from one or a small number of memories, and can clearly correspond to content, time, location, reason, purpose, or keywords.
Suitable for questions about facts, events, time, location, reasons, purposes, people, objects, names, profile content, etc.

hybrid:
The question has a clear retrieval target, but the answer requires aggregation, counting, comparison, sorting, calculation, or reasoning over multiple memories.
Suitable for questions involving how many times, total, average, difference, percentage, more/less than, before/after, earliest/latest, first/last/order, based on, etc.

raw:
The question is not suitable for reliable structured parsing and mainly relies on query_anchor and keywords for broad BM25 + embedding recall.
Suitable for recommendations, suggestions, open-ended planning, vague recall, strong context dependence, or questions that cannot be clearly constrained structurally.
If forcibly filling structured fields would introduce incorrect constraints, choose raw.

2. query_anchor

A natural language retrieval query rewritten from the original question. It is not a keyword list.

Requirements:
- Preserve the core intent of the original question.
- Preserve the person names, objects, events, and key context in the original question.
- Do not change specific subject names.
- You may moderately complete omitted information to make the query more suitable for retrieval.
- Preserve important time, location, comparison, quantity, and order information.

Examples:

Why did Sophie visit the design studio? -> What was Sophie's purpose for visiting the design studio?

What recipe did Nathan try after the cooking workshop? -> What recipe did Nathan try after attending the cooking workshop?

Which museum would Clara likely enjoy visiting based on her interest in modern sculpture? -> Which museum would Clara likely enjoy visiting based on Clara's interest in modern sculpture?

3. dimension.target_memory_type

Indicates the memory types to prioritize for retrieval. Multiple values may be selected. If unclear or if retrieval mainly uses raw mode, use [].

Allowed values:
- fact: Stable facts, identity, background, relationships, current status, tools, models, datasets, configurations, possessions, or stable attributes.
- episodic: Specific events, experiences, actions, purchases, trips, plans, phase progress, or timeline-related memories.
- profile: Preferences, habits, interests, values, long-term goals, ability traits, style preferences, or stable behavioral patterns.

4. dimension.keywords

Key retrieval terms or phrases in the question, used for recall, deduplication, and query-memory alignment.

Extraction focus:
- fact: People, objects, tools, models, datasets, projects, relationship objects, stable attributes.
- episodic: Events, participants, locations, activities, key objects, outcomes, specific actions.
- profile: Preference objects, habitual activities, interest domains, value objects, long-term goal objects.

Rules:
- Use short words or noun phrases.
- Preserve important person names.
- May include important entities, topic words, action words, or object words.
- Do not include full sentences.
- Do not duplicate keywords.
- Do not extract common words with no retrieval value.
- Use [] when there are no clear keywords.
- Even when parse_mode is raw, still extract keywords as much as possible.

5. dimension.time

Represents the time constraint in the question, used to match memory.dimension.time.

Fill this field only when the question itself contains an explicit time constraint.

Allowed formats only:
- "on <specific time>"
- "before <specific time>"
- "after <specific time>"
- "around <specific time>"
- "between <start time> and <end time>"
- ""

Time expressions should be as comparable as possible, for example:
- "2023-05-08"
- "2023-05"
- "2023"
- "May 8, 2023"
- "March"

Rules:
- If the question contains an explicit date, month, year, or comparable time range, fill in time.
- If the question is asking about time, time is not a constraint and answer_dim should be set to "time".
- If there is no question_date, do not normalize relative time expressions such as today, yesterday, this month, last week, currently, now, recently.
- currently, now, and recently usually indicate current status or recent status, and should not be written into time.
- Do not write frequencies such as every Saturday, usually, often, twice a week, daily, weekly into time.
- Do not write latest, earliest, first, or previous alone into time. These usually belong to hybrid mode.
- If there is no explicit time constraint, use "".

Examples:

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

Represents the location, platform, scenario, or environment constraint in the question, used to match memory.dimension.location.

Rules:
- If a location, platform, or scenario is a retrieval condition, fill in location.
- If the question is asking about location, location is not a constraint and answer_dim should be set to "location".
- Do not force ordinary topics into location.
- If there is no location constraint, use "".

Examples:

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

Indicates the memory field that the answer directly corresponds to.

Allowed values only:
- "content"
- "time"
- "location"
- "reason"
- "purpose"
- "keywords"
- ""

Filling rules:
- For ordinary facts, event content, or profile content: content
- For time, dates, months, weekdays, frequencies, or usual times: time
- For locations, platforms, or scenarios: location
- For reasons: reason
- For purposes, uses, or goals: purpose
- For people, objects, organizations, names, targets, platforms, works, or other key phrases: keywords
- If the question requires counting, summation, calculation, comparison, sorting, yes/no judgment, recommendation, inference, summarization, or complex reasoning, use "".

The following is the actual question to parse:

Question:
{{question}}
"""