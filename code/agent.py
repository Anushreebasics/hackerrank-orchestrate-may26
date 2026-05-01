import json
import time
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

class TriageOutput(BaseModel):
    chain_of_thought: str = Field(description="Internal scratchpad: step-by-step reasoning used to arrive at the final decision.")
    status: str = Field(description="Must be exactly 'replied' or 'escalated'")
    product_area: str = Field(description="A short string categorizing the domain area based on the context.")
    response: str = Field(description="A helpful, user-facing answer if replying. If escalating, provide a brief, professional explanation.")
    justification: str = Field(description="Internal reasoning explaining WHY you chose to reply/escalate and how the corpus supports it.")
    request_type: str = Field(description="Must be exactly 'product_issue', 'feature_request', 'bug', or 'invalid'")


def is_valid_company(company: str) -> bool:
    """Check if company is valid (not None, not 'None', not empty)."""
    if not company or company.strip().lower() in ("none", ""):
        return False
    return True


def is_gratitude(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    # common short gratitude phrases
    gratitude_phrases = [
        "thank you",
        "thanks",
        "thankyou",
        "thanks!",
        "thank you!",
        "thank you for helping me",
        "thank you for your help",
        "thanks for your help",
        "thanks!"
    ]
    # match exact short gratitude or very short lines
    if t in gratitude_phrases:
        return True
    # if text is very short and contains 'thank' or 'thanks'
    if len(t.split()) <= 3 and ("thank" in t or "thanks" in t):
        return True
    return False

class SupportAgent:
    def __init__(self):
        self.client = genai.Client()
        self.last_call = 0
        
    def generate_response(self, issue, subject, company, retrieved_chunks):
        # If the user is just saying thanks, reply courteously without requiring company
        if is_gratitude(issue):
            return TriageOutput(
                chain_of_thought="Step 1: Read the user issue. Step 2: Detect short gratitude message. Step 3: No product context required; reply with a polite acknowledgement.",
                status="replied",
                product_area="General",
                response="You're welcome — happy to help!",
                justification="Detected a gratitude message; no triage or escalation required.",
                request_type="invalid"
            )

        # Immediate escalation if company is missing or invalid
        if not is_valid_company(company):
            return TriageOutput(
                chain_of_thought="Step 1: Check company field. Step 2: Company is missing or 'None'. Step 3: Cannot reliably determine product scope without company context. Step 4: Escalate.",
                status="escalated",
                product_area="Unknown",
                response="Unable to determine which product this issue belongs to. Please resubmit your ticket with the product/company name (HackerRank, Claude, or Visa).",
                justification="Company field is missing or empty. Cannot triage without knowing which product context to use.",
                request_type="invalid"
            )
        
        context_str = "\n\n---\n\n".join([f"Source ({doc['company']} - {doc['filepath']}):\n{doc['text']}" for doc in retrieved_chunks])
        
        system_instruction = """You are an expert technical support triage agent for HackerRank, Claude, and Visa.
Your goal is to evaluate incoming support tickets and either provide a safe, grounded response or escalate the ticket.

RULES:
1. You MUST rely ONLY on the provided support corpus context. Do not use outside knowledge. Do not hallucinate policies or links.
2. If the context does NOT contain the answer, or the issue is high-risk, sensitive, or asks for actions outside your scope, you MUST escalate.
3. If the request is malicious, irrelevant, or inappropriate, set request_type to "invalid" and escalate.
4. Output must be perfectly structured according to the required schema.
            
SELF-REFLECTION / CHAIN-OF-THOUGHT:
- Provide a detailed, step-by-step internal scratchpad in the `chain_of_thought` field BEFORE you set `status`.
- Example: "Step 1: Read the user issue. Step 2: Examine retrieved context for matching phrases. Step 3: Determine if an exact solution exists in context. Step 4: If found, prepare reply; otherwise escalate."
- This field is for internal reasoning ONLY. Keep it factual, numbered, and concise.
"""

        prompt = f"""
Ticket Information:
- Subject: {subject}
- Company: {company}
- Issue: {issue}

Retrieved Context from Support Corpus:
{context_str}

Evaluate the ticket based strictly on the retrieved context above.
"""
        
        try:
            # Respect 5 RPM free limit (1 request every 12 seconds)
            elapsed = time.time() - self.last_call
            if elapsed < 13.0:
                time.sleep(13.0 - elapsed)
            
            # Retry with exponential backoff for 503 errors
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    response = self.client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            response_mime_type="application/json",
                            response_schema=TriageOutput,
                            temperature=0.0,
                        ),
                    )
                    self.last_call = time.time()
                    data = json.loads(response.text)
                    return TriageOutput(**data)
                except Exception as e:
                    error_str = str(e)
                    if "503" in error_str and attempt < max_retries - 1:
                        print(f"API 503 error on attempt {attempt + 1}, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        raise
                        
        except Exception as e:
            print(f"LLM API error (returning fallback): {e}")
            return TriageOutput(
                chain_of_thought=f"LLM error after retries: {type(e).__name__}",
                status="escalated",
                product_area="Unknown",
                response="Error processing request. Please resubmit your ticket.",
                justification=f"LLM service error: {type(e).__name__}",
                request_type="invalid"
            )
