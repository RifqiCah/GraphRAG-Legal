LEGAL_REFERENCE_EXTRACTION_PROMPT = """
You are an expert legal evaluator. Your task is to extract all legal references mentioned in the text.
A legal reference typically includes the regulation type (UU, PP, Permendikbud, etc.), number, year, and optionally the article (Pasal) and paragraph (ayat).

Given a text, extract a list of all distinct legal references mentioned in it.
Provide the original mentioned text for each reference. If no legal references are mentioned, return an empty list for 'references'.

Provide your response in strictly valid JSON format matching this schema:
{{
  "references": [
    {{
      "mentioned_reference": "<the exact substring from the text mentioning the reference>"
    }}
  ]
}}

Text:
{text}
"""

LEGAL_SUPPORT_CHECK_PROMPT = """
You are an expert legal evaluator. Your task is to check if a specific legal text supports the claim made in a generated answer.
You will be provided with:
1. A generated answer containing a claim.
2. The specific legal reference mentioned in that answer.
3. The actual text of that legal reference from the official corpus.

Evaluate whether the actual text of the legal reference supports the claim made in the generated answer.
Provide a boolean "supports_claim", a brief "evidence" quote from the actual text, and a "reason".

Provide your response in strictly valid JSON format matching this schema:
{{
  "supports_claim": true | false,
  "evidence": "<exact quote from actual text or null>",
  "reason": "<short explanation>"
}}

Generated Answer:
{generated_answer}

Mentioned Reference:
{mentioned_reference}

Actual Legal Text from Corpus:
{actual_text}
"""

CONTEXT_REFERENCE_CHECK_PROMPT = """
You are an expert legal evaluator. Your task is to verify if a specific legal reference is present in the provided retrieved context.
You will be given:
1. A legal reference (e.g. "UU No 20 Tahun 2003 Pasal 5").
2. The retrieved context (a potentially long text).

Evaluate whether the context contains the text or substance of the specified legal reference.
Return a boolean "is_present", a brief "evidence" quote from the context if found, and a "reason".

Provide your response in strictly valid JSON format matching this schema:
{{
  "is_present": true | false,
  "evidence": "<exact quote from context or null>",
  "reason": "<short explanation>"
}}

Target Legal Reference:
{target_reference}

Retrieved Context:
{context}
"""
