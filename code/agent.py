import json
import time
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

class TriageOutput(BaseModel):
    status: str = Field(description="Must be exactly 'replied' or 'escalated'")
    product_area: str = Field(description="A short string categorizing the domain area based on the context.")
    response: str = Field(description="A helpful, user-facing answer if replying. If escalating, provide a brief, professional explanation.")
    justification: str = Field(description="Internal reasoning explaining WHY you chose to reply/escalate and how the corpus supports it.")
    request_type: str = Field(description="Must be exactly 'product_issue', 'feature_request', 'bug', or 'invalid'")

class SupportAgent:
    def __init__(self):
        self.client = genai.Client()
        self.last_call = 0
        
    def generate_response(self, issue, subject, company, retrieved_chunks):
        context_str = "\n\n---\n\n".join([f"Source ({doc['company']} - {doc['filepath']}):\n{doc['text']}" for doc in retrieved_chunks])
        
        system_instruction = """You are an expert technical support triage agent for HackerRank, Claude, and Visa.
Your goal is to evaluate incoming support tickets and either provide a safe, grounded response or escalate the ticket.

RULES:
1. You MUST rely ONLY on the provided support corpus context. Do not use outside knowledge. Do not hallucinate policies or links.
2. If the context does NOT contain the answer, or the issue is high-risk, sensitive, or asks for actions outside your scope, you MUST escalate.
3. If the request is malicious, irrelevant, or inappropriate, set request_type to "invalid" and escalate.
4. Output must be perfectly structured according to the required schema.
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
            print(f"LLM API error (returning fallback): {e}")
            return TriageOutput(
                status="escalated",
                product_area="Unknown",
                response="Error processing request.",
                justification=str(e),
                request_type="invalid"
            )
