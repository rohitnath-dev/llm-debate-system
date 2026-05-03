import chromadb
import uuid
from dotenv import load_dotenv
import os
from openai import OpenAI
import json
import re

load_dotenv()

def init_client():
    default_base_url = "https://openrouter.ai/api/v1"
    default_model = "deepseek/deepseek-chat"   # free + stable

    print("\n(Press Enter to use defaults)")

    api_key = input("Enter API key (OpenRouter recommended): ").strip()

    base_url = input(f"Base URL [{default_base_url}]: ").strip()
    if not base_url:
        base_url = default_base_url

    model = input(f"Model [{default_model}]: ").strip()
    if not model:
        model = default_model

    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    return client, model

class MultiLLM:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def _call(self, system_prompt, user_prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"{system_prompt}\n\n{user_prompt}"
            }],
            temperature=0.3,
            timeout=20
        )
        return response.choices[0].message.content

    def responder(self, system_prompt, user_prompt):
        return self._call(system_prompt, user_prompt)

    def critic(self, system_prompt, user_prompt):
        return self._call(system_prompt, user_prompt)

    def judge(self, system_prompt, user_prompt):
        return self._call(system_prompt, user_prompt)
        
# Short Term Memory (STM)
stm = []
MAX_HISTORY = 5

def update_stm(role, content):
    stm.append({"role": role, "content": content})

    if len(stm) > MAX_HISTORY:
        stm.pop(0)
        
# Long Term Memory (LTM)
chroma_client = chromadb.Client()
memory_collection = chroma_client.create_collection(name="memory")

def store_memory(text):
    chunks = text.split("\n")  # split by bullet / lines

    for chunk in chunks:
        chunk = chunk.strip()

        if chunk:  # ignore empty lines
            memory_collection.add(
                documents=[chunk],
                ids=[str(uuid.uuid4())]
            )
            
def retrieve_memory(query, k=2):
    results = memory_collection.query(
        query_texts=[query],
        n_results=k
    )

    if not results["documents"]:
        return []

    return results["documents"][0]
    
def build_context(query):
    stm_block = "\n".join([f"{m['role']}: {m['content']}" for m in stm])

    ltm = retrieve_memory(query)
    ltm_block = "\n".join(ltm) if ltm else ""

    return f"""
Relevant long-term memory:
{ltm_block}

Recent conversation:
{stm_block}

Current query:
{query}
"""
    
memory_prompt = """
Extract useful long-term memory from the user input.

Only include:
- stable facts
- preferences
- identity
- goals

Ignore:
- greetings
- temporary info
- one-time questions

Return:
- "NONE" if nothing useful
- otherwise short bullet points

User input:
"""

def extract_memory(user_input):
    raw = llm.responder(memory_prompt, user_input)

    if raw.strip().upper() == "NONE":
        return None

    return raw
    
responder_prompt = '''
You generate answers.

Inputs:
- user_query

Rules:
- First understand the core issue.
- Give direct, logical, actionable advice.
- Avoid vague or motivational language.
- Break complex problems into simple steps when needed.

Response length (strict):
- If the query is short → respond in 1–3 concise sentences.
- If the query is detailed → give a structured, step-by-step answer.
- Do not over-explain simple queries.
- Do not under-explain complex queries.

Use the user's name and relevant details from memory (STM and LTM) when available to personalize the response

Output format (strict JSON):
{
  "answer": "response",
  "stance": "DEFENDED | CHALLENGED"
}
'''

improve_prompt = '''
You refine an existing answer.

Inputs:
- user_query
- current_answer
- latest_critique

Rules:
- Improve clarity, logic, and completeness.
- Fix real issues mentioned in the critique.
- Do not blindly accept critique.
- Reject incorrect or weak critique points.
- Keep the answer concise and direct.
- Do not change the core meaning unless necessary.

- First understand the core issue.
- Give direct, logical, actionable advice.
- Break complex ideas into simple steps when needed.

Response length (strict):
- If the query is short → respond in 1–3 concise sentences.
- If the query is detailed → give a structured, step-by-step answer.
- Do not over-explain simple queries.
- Do not under-explain complex queries.

Output format (strict JSON):
{
  "answer": "refined answer",
  "stance": "MODIFIED | DEFENDED"
}
'''

critic_prompt = '''
You are a strict critic.

Inputs:
- user_query
- current_answer

Rules:
- Point out flaws, gaps, weak reasoning, missing details
- Be direct and argumentative
- Do not rewrite full answer
- If answer is mostly correct, acknowledge briefly but still test it
- If no major issue → say it is acceptable but mention minor improvement if any
- Avoid repeating same points

If the answer is correct, DO NOT criticize tone, friendliness, or style.
Only point out factual or logical issues.

Do not suggest unnecessary improvements.

Critique policy:
- Focus ONLY on meaningful issues (logic, correctness, relevance).
- Ignore punctuation, tone, or stylistic preferences.
- Do NOT nitpick minor details like punctuation or wording.
- If the answer is correct and appropriate, say: "No major issues".

Output format (strict JSON):
{
  "critique": "direct critique text",
  "severity": "LOW | MEDIUM | HIGH"
}
'''

judge_prompt = '''
You evaluate whether a critique is worth applying.

Inputs:
- user_query
- current_answer
- latest_critique

Rules:
- Focus only on correctness, relevance, and clarity.
- Ignore tone, style, wording, punctuation.
- Useful critique = identifies real issue (logic, facts, relevance).
- Not useful = nitpicking, overthinking, or unnecessary changes.

Special case:
- If query is a simple greeting (e.g., "hi", "hello"):
  → Any valid greeting response is sufficient
  → use_critique = false

Decision:
- If critique improves answer meaningfully → use_critique = true
- Otherwise → use_critique = false

Output (strict JSON):
{
  "use_critique": true/false,
  "confidence": 0-1,
  "reason": "short reason"
}'''

current_answer = ""
latest_critique = ""

def extract_json(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError("No valid JSON found")
        
def run_responder():
    global current_answer, parsed_responder

    context = build_context(user_prompt)
    raw = llm.responder(responder_prompt, context)

    try:
        data = extract_json(raw)
        parsed_responder = data
        current_answer = data.get("answer", "")
    except Exception as e:
        print("[ERROR] Responder failed:", e)
        
def run_critic():
    global latest_critique, parsed_critic

    critic_input = f"""
{build_context(user_prompt)}
current_answer: {current_answer}
"""

    raw = llm.critic(critic_prompt, critic_input)

    try:
        data = extract_json(raw)
        parsed_critic = data
        latest_critique = data.get("critique", "")
    except Exception as e:
        latest_critique = ""
        print("[ERROR] Critic failed:", e)
        
def run_improve():
    global current_answer, parsed_responder

    improve_input = f"""
{build_context(user_prompt)}

current_answer: {current_answer}
latest_critique: {latest_critique}
"""

    raw = llm.responder(improve_prompt, improve_input)

    try:
        data = extract_json(raw)
        parsed_responder = data
        current_answer = data.get("answer", current_answer)
    except Exception as e:
        print("[ERROR] Improve failed:", e)
        
def run_judge():
    global judge_output, parsed_judge

    judge_input = f"""
{build_context(user_prompt)}
current_answer: {current_answer}
latest_critique: {latest_critique}
"""

    raw = llm.judge(judge_prompt, judge_input)

    try:
        data = extract_json(raw)
        parsed_judge = data
        judge_output = data
    except Exception as e:
        judge_output = {"use_critique": False}
        print("[ERROR] Judge failed:", e)
        
        
if __name__ == "__main__":

    client, model = init_client()
    llm = MultiLLM(client, model)

    while True:
        user_prompt = input("\n> ")

        if user_prompt.lower() in ["exit", "quit"]:
            print("\nSession ended.")
            break

        current_answer = ""
        latest_critique = ""
        judge_output = {"use_critique": True}

        update_stm("user", user_prompt)

        print("\nThinking...\n")

        history = []

        for i in range(3):

            step = {}

            # --- Responder / Improve ---
            if i == 0:
                run_responder()
                step["stage"] = "Initial Answer"
            else:
                run_improve()
                step["stage"] = "Refined Answer"

            step["answer"] = current_answer

            # --- Critic ---
            run_critic()
            step["critique"] = latest_critique
            step["severity"] = parsed_critic.get("severity") if "parsed_critic" in globals() else None

            # --- Judge ---
            run_judge()
            step["use_critique"] = judge_output.get("use_critique")
            step["reason"] = parsed_judge.get("reason") if "parsed_judge" in globals() else None

            history.append(step)

            if not judge_output.get("use_critique", True):
                break

        # ===== DISPLAY (clean, user-friendly) =====

        print("=== Conversation Breakdown ===\n")

        for i, step in enumerate(history):
            print(f"Step {i+1}: {step['stage']}")
            print(f"Answer: {step['answer']}")

            if step["critique"]:
                print(f"Critic: {step['critique']} ({step['severity']})")

            print(f"Judge: {'Accepted critique' if step['use_critique'] else 'No improvement needed'}")

            if step["reason"]:
                print(f"Reason: {step['reason']}")

            print("-" * 40)

        print("\n=== Final Answer ===\n")
        print(current_answer)

        update_stm("assistant", current_answer)

        memory = extract_memory(user_prompt)
        if memory:
            store_memory(memory)
