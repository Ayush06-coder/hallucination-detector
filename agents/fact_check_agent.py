from langchain_groq import ChatGroq
from ddgs import DDGS
from ddgs.exceptions import DDGSException, TimeoutException
from dotenv import load_dotenv

import os
import time


load_dotenv()


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
    temperature=0
)


# ---------------------------------------------------------
# WEB SEARCH
# ---------------------------------------------------------

def search_web(query):
    """
    Search the web for factual evidence.

    Uses retry/fallback handling so temporary
    network/search failures do not crash the detector.

    Returns:
        str: Search evidence or an explicit
             WEB_SEARCH_UNAVAILABLE message.
    """

    print("🌐 Searching web for evidence...")

    search_configs = [
        {
            "verify": True,
            "label": "normal SSL verification"
        },
        {
            "verify": False,
            "label": "SSL fallback"
        }
    ]

    last_error = None

    for attempt_number, config in enumerate(
        search_configs,
        start=1
    ):

        try:

            print(
                f"   Search attempt {attempt_number} "
                f"({config['label']})"
            )

            with DDGS(
                timeout=10,
                verify=config["verify"]
            ) as ddgs:

                results = list(
                    ddgs.text(
                        query,
                        region="us-en",
                        safesearch="moderate",
                        max_results=5,
                        backend="auto"
                    )
                )

            if not results:

                print(
                    "   ⚠️ Search returned no results."
                )

                if attempt_number < len(search_configs):
                    time.sleep(1)

                continue

            evidence_parts = []
            seen_text = set()

            for result in results:

                title = result.get(
                    "title",
                    ""
                ).strip()

                body = result.get(
                    "body",
                    ""
                ).strip()

                href = result.get(
                    "href",
                    ""
                ).strip()

                if not body:
                    continue

                normalized_body = body.lower()

                if normalized_body in seen_text:
                    continue

                seen_text.add(normalized_body)

                if len(body) > 700:
                    body = body[:700] + "..."

                if title:

                    source_block = (
                        f"Source: {title}\n"
                        f"Evidence: {body}"
                    )

                else:

                    source_block = (
                        f"Evidence: {body}"
                    )

                if href:

                    source_block += (
                        f"\nURL: {href}"
                    )

                evidence_parts.append(
                    source_block
                )

            if evidence_parts:

                evidence = "\n\n".join(
                    evidence_parts
                )

                print(
                    f"   ✅ Web search successful "
                    f"({len(evidence_parts)} useful results)"
                )

                return evidence

            print(
                "   ⚠️ Search results contained "
                "no usable evidence."
            )

        except (
            DDGSException,
            TimeoutException
        ) as e:

            last_error = e

            print(
                f"   ⚠️ Search attempt "
                f"{attempt_number} failed: {e}"
            )

            if attempt_number < len(search_configs):

                print("   🔄 Retrying...")

                time.sleep(1)

        except Exception as e:

            last_error = e

            print(
                f"   ⚠️ Unexpected search error: {e}"
            )

            if attempt_number < len(search_configs):

                print("   🔄 Retrying...")

                time.sleep(1)

    print("❌ Web search unavailable.")

    if last_error:

        print(
            f"   Last error: {last_error}"
        )

    return (
        "WEB_SEARCH_UNAVAILABLE: "
        "External web search could not be completed. "
        "Do not treat this as supporting evidence."
    )


# ---------------------------------------------------------
# FACT-CHECK AGENT
# ---------------------------------------------------------

def fact_check_agent(
    question,
    llm_response
):

    print("🔍 Fact-Check Agent running...")

    # Direct factual queries generally produce
    # better search results.
    search_query = question.strip()

    evidence = search_web(
        search_query
    )

    prompt = f"""
You are a strict fact-checking agent in a
multi-agent hallucination detection system.

Your job is to determine whether the AI-generated
answer is factually supported by the available
external evidence.

ORIGINAL QUESTION:
{question}

AI-GENERATED ANSWER:
{llm_response}

EXTERNAL WEB EVIDENCE:
{evidence}


=========================================================
CLAIM INTERPRETATION
=========================================================

Before deciding the verdict, identify:

1. The exact factual claim made by the AI answer.

2. The entity, person, place, object, event, or concept
   referred to by the claim.

3. The context in which the entity is being discussed.

4. Whether the external evidence refers to the SAME
   entity and SAME context.

5. Whether the evidence actually supports or contradicts
   the claim.

Do NOT assume similarly named entities are the same.

For example, "Atlantis" could refer to:

- the mythical city described by Plato
- Atlantis, Florida
- another location, organization, or entity

If the question does not provide enough information to
determine the intended entity, treat that ambiguity as
important evidence and prefer UNCERTAIN.

Do not silently choose one interpretation simply because
search results happen to contain that interpretation.


=========================================================
EVIDENCE INTERPRETATION
=========================================================

Evaluate the relationship between the claim and evidence.

Evidence can:

- directly support the claim
- directly contradict the claim
- partially support the claim
- provide an estimate
- provide a range
- use different wording
- be irrelevant
- refer to a different entity
- be insufficient


=========================================================
NUMBERS, RANGES AND PRECISION
=========================================================

Be especially careful with numerical claims.

Example:

CLAIM:
"The population is exactly 5 million."

EVIDENCE:
"The estimated population is between 5 and 6 million."

This does NOT automatically prove the claim FALSE.

The evidence is compatible with 5 million being within
the estimated range.

Therefore, unless the evidence explicitly establishes
that the exact value cannot be 5 million, prefer:

UNCERTAIN

rather than:

FALSE.

Similarly:

CLAIM:
"The population is exactly 5 million."

EVIDENCE:
"The population is exactly 2,142."

If both refer to the SAME entity and SAME time/context,
this is a clear contradiction and should be:

FALSE.

Do not treat:

- estimates as exact values
- ranges as exact values
- approximate numbers as precise measurements
- different dates as directly comparable
- different contexts as equivalent

=========================================================
SOURCE RELIABILITY
=========================================================

Not all search results are equally reliable.

When evaluating external evidence, consider the
reliability of the source.

Prefer evidence from:

- Government websites
- Universities and academic institutions
- National libraries
- Museums and established research institutions
- Official organizations
- Well-established reference works
- Reputable news organizations

Treat evidence from:

- Personal blogs
- Forums
- Social media
- Unsourced websites
- SEO/content-farm websites
- Anonymous webpages

as lower-reliability evidence.

IMPORTANT:

Do not automatically mark a claim UNCERTAIN just because
one low-quality source disagrees with a high-quality source.

When sources conflict:

1. Identify the reliability of each source.
2. Prefer authoritative and well-supported sources.
3. Determine whether multiple independent sources support
   the same conclusion.
4. Only return UNCERTAIN when the conflict remains
   genuinely unresolved after considering source quality.

A single weak contradictory source should not outweigh
multiple authoritative sources.

For historical or scientific claims, prioritize sources
from recognized institutions and established references.

=========================================================
VERDICT RULES
=========================================================

Return TRUE when:

- The evidence clearly supports the claim, AND
- the evidence refers to the same entity/context, AND
- there is no meaningful contradiction.

Return FALSE when:

- The evidence clearly contradicts the claim, AND
- the evidence refers to the same entity/context, AND
- the contradiction is strong enough to justify rejecting
  the claim.

Return UNCERTAIN when:

- Evidence is insufficient.
- Evidence is ambiguous.
- Evidence only partially addresses the claim.
- The entity is ambiguous.
- Sources conflict.
- The claim cannot be reliably verified.
- The evidence refers to a different entity.
- The evidence refers to a different time/context that
  prevents a reliable comparison.
- The difference is only a matter of precision.
- The evidence provides a range containing the claimed
  value.
- The evidence provides an estimate rather than an exact
  value.
- The evidence does not fully answer the question.


=========================================================
IMPORTANT
=========================================================

Do NOT mark a claim FALSE merely because:

- the evidence uses a range instead of an exact number
- the evidence gives an estimate
- the evidence uses slightly different wording
- the evidence refers to a different entity
- the evidence refers to a different time period
- the evidence does not fully answer the question
- the evidence is less precise than the AI answer

Do NOT mark a claim TRUE merely because:

- the answer sounds plausible
- the evidence is vaguely related
- the search result contains similar words
- the evidence refers to a different entity
- the evidence does not actually establish the claim


=========================================================
SEARCH FAILURE
=========================================================

If the external evidence begins with:

WEB_SEARCH_UNAVAILABLE

you MUST NOT treat that text as evidence.

Return:

VERDICT: UNCERTAIN

unless the information already supplied in the question
and answer is sufficient for a reliable determination.


=========================================================
SOURCE QUALITY
=========================================================

Do not blindly trust one search result.

Consider:

- whether multiple results agree
- whether the source appears relevant
- whether the source actually supports the claim
- whether the source is discussing the same entity
- whether the source is discussing the same date/context

If search results conflict and the conflict cannot be
resolved reliably, return UNCERTAIN.


=========================================================
FINAL INSTRUCTIONS
=========================================================

- Never invent evidence.
- Never invent sources.
- Never assume missing information.
- Never blindly trust the AI-generated answer.
- Never blindly trust a single search snippet.
- Compare the CLAIM directly with the EVIDENCE.
- Prefer UNCERTAIN over an unjustified TRUE or FALSE.
- Be conservative when evidence is ambiguous.


Reply ONLY in this exact format:

VERDICT: [TRUE/FALSE/UNCERTAIN]
REASON: [one concise sentence explaining the decision]
EVIDENCE: [one concise factual statement from the evidence, or "No reliable external evidence available."]
"""

    try:

        response = llm.invoke(
            prompt
        )

        content = response.content.strip()

        print(
            "   ✅ Fact-check completed."
        )

        return content

    except Exception as e:

        print(
            f"❌ Fact-Check Agent failed: {e}"
        )

        return """
VERDICT: UNCERTAIN
REASON: Fact-checking could not be completed because the verification service failed.
EVIDENCE: No reliable external evidence available.
"""


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    question = (
        "Who invented the telephone?"
    )

    llm_response = (
        "Alexander Graham Bell invented "
        "the telephone in 1876."
    )

    result = fact_check_agent(
        question,
        llm_response
    )

    print("\n" + "=" * 60)
    print(result)
    print("=" * 60)