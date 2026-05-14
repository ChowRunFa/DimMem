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

The first {overlap_count} messages in the input, namely messages numbered 1 to {overlap_count}, are overlapping context from the previous conversation segment. They are mainly used to understand message {extract_start_index} and later messages.

By default, do not extract memories from the first {overlap_count} messages, unless message {extract_start_index} and later must rely on them to form a complete and self-contained memory.
"""

LOCOMO_QUERY_ANALYSIS_PROMPT = """
You are a memory query parser. Convert natural language questions into structured retrieval queries. Output only valid JSON.

== Output Format ==

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

== Field Descriptions ==

1. query_anchor
Rewrite the original question into a retrieval-friendly natural language sentence. I/me/my -> the user. Preserve key information such as the core intent, time, location, quantity, and order. This is not a keyword list.
Example true: "Can you remind me which airline you suggested last time for budget flights?"
Example false: "How many countries have I visited this year?"

2. dimension.target_memory_type
Memory types to prioritize for retrieval; multiple values are allowed:
- fact: stable facts, identity, relationships, status, possessions
- episodic: specific events, experiences, actions, purchases, trips, progress
- profile: preferences, habits, interests, goals, style
Use [] when uncertain.

3. dimension.keywords
Extract phrases for key entities, people, objects, tools, locations, activities, topics, etc. Use [] if none.

4. dimension.time
Fill only when there is an explicit time constraint. Format: "on/before/after/around <time>" or "between <start> and <end>".
- If question_date is available, normalize relative time expressions (today/yesterday/this week/last month, etc. -> specific dates)
- If the question asks about the time itself, leave this field empty and set answer_dim = "time"
- Frequency words (daily/weekly/often) are not time constraints
- Use "" when there is no explicit constraint

5. dimension.location
Fill only when there is an explicit location/platform/scene constraint. If the question asks about the location itself, leave this field empty and set answer_dim = "location". Use "" if none.

6. answer_dim
The memory field corresponding to the answer:
- "content": fact/event/profile content
- "time": time/date/frequency
- "location": location/platform/scene
- "reason": reason
- "purpose": purpose/usage
- "keywords": key objects such as people/objects/names/tools
- "": requires calculation/comparison/ranking/reasoning/recommendation/summarization

== Input ==

Question:
{question}
"""