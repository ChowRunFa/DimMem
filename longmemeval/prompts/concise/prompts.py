LONGMEMEVAL_STRUCTURED_MEMORY_EXTRACTION_PROMPT = """
You are a structured memory extractor.

Task: Extract structured memories with long-term value from the input user-assistant conversation, and strictly output valid JSON.
Only output JSON. Do not output explanations, analysis, Markdown, or any extra text.

========================
Output Format
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
Extraction Targets
========================

Extract:
1. Factual information, identity/background, relationships, current status, tools, models, datasets, and project configurations;
2. Specific experiences, events, actions, behavioral records, stage progress, and future plans;
3. Long-term preferences, habits, interests, values, goals, abilities, interaction style, or writing style;
4. Information that helps understand the user's future needs or retrieve the user's background.

Do not extract:
1. Greetings, thanks, simple confirmations, or meaningless small talk;
2. Temporary formatting requirements, one-off operation instructions, or current-task details without long-term value.

========================
memory_type
========================

dimension.memory_type must be one of: fact, episodic, profile.

fact: stable facts, answering "what it is / what exists / what is used / what the relationship is / what the current status is".
Includes identity, background, relationships, status, tools, models, datasets, configurations, confirmed choices, and stable objective attributes.
Example: The user uses LLaMA2-7B as the base model.

episodic: specific events, answering "what happened / what someone did / what someone experienced / what someone plans to do".
Includes a specific event, experience, action, stage progress, future plan, or concrete fact with time/location/context.
Example: The user plans to train a local LLaMA2-7B model using Urdu data.

profile: long-term user profile, answering "what someone is like long-term / what someone likes / what someone usually does / what someone believes".
Includes preferences, habits, interests, values, long-term goals, abilities, style preferences, and stable behavior patterns.
Example: The user prefers concise Java code with a single main function.

Do not extract content that cannot be classified as fact, episodic, or profile.

========================
content Rules
========================

content is the main text of the memory and must be one complete, self-contained, retrievable sentence.

Requirements:
1. Clearly state who the memory is about and the core fact, event, or profile information.
2. If the source text contains time, location, reason, or purpose, include them in content when possible.
3. Remove ambiguous pronouns so that content does not depend on the original context.
4. Normalize relative time expressions based on the message timestamp, e.g., yesterday → a specific date.
5. Do not add unsupported information, and do not overgeneralize a single event into a long-term profile.

========================
dimension Rules
========================

dimension is used for structured retrieval. Except for memory_type, use "" when there is no clear evidence; use [] for keywords when there are no clear keywords.

time: the time when the memory is valid, happened, is planned to happen, or repeatedly occurs.
Use an absolute date if available. Normalize relative time if the message timestamp is available. Use "" if there is no time.
Do not use the current system time unless it is the message timestamp.

location: physical place, online platform, organizational context, home space, workplace, system environment, or activity venue.
Fill this only when the source text explicitly mentions or strongly implies it. Do not force ordinary topics into location.

reason: cause, motivation, trigger, or background condition.
Fill this only when the source text explicitly states or strongly implies it. Do not infer hidden motivations. Do not confuse it with purpose.

purpose: goal, intention, or expected outcome.
Fill this only when the source text explicitly states or strongly implies it. Do not infer unstated purposes.

keywords: key terms or phrases for retrieval, deduplication, and query-memory alignment.
Extract subjects, objects, tools, models, datasets, projects, people, locations, activities, results, preference objects, interest domains, etc.
Keywords must be short words or noun phrases. Do not include full sentences. Do not repeat keywords. Do not extract ordinary words without retrieval value.

========================
Extraction Rules
========================

1. Process messages in chronological order.
2. Extract memories mainly from user messages.
3. Each memory should be as atomic as possible. If a message is long or contains multiple independent information points, split it into multiple memories, preserving key details such as people, time, location, events, reasons, purposes, preferences, and objects separately. Avoid over-merging or omitting details.
4. The `content` field must be self-contained and must not rely on the original dialogue context.
5. The `time` field should be normalized based on the message timestamp: relative time expressions must be converted into absolute dates or time ranges. For example, if the message timestamp is `2023-05-08` and the original text says `yesterday`, then `time` should be `2023-05-07`.
6. Simple confirmations, temporary formatting requirements, and one-off tasks in the current conversation should generally not be extracted.
7. The output must be valid JSON. Do not output any text outside the JSON.

{overlap_rule}

Here is the real input you need to process:

{conversation}
"""


OVERLAP_RULE = '''
========================
Input and Overlap Context Rules
========================

You will receive a "current conversation segment".
The first {overlap_count} messages in the input, namely messages numbered 1 to {overlap_count}, are overlapping context from the previous segment and must not be used as new memory sources.
The first {overlap_count} messages may only be used to understand later messages; the content that is actually allowed for extraction starts from message {extract_start_index}.
'''
