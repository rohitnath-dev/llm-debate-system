# llm-debate-system

Experimental multi-agent LLM system using a responder–critic–judge loop for iterative reasoning. Currently a basic prototype, with planned upgrades including memory (short/long term), multi-answer generation, and more advanced reasoning workflows.

## What this is

This project is a simple but practical attempt to move beyond single-shot LLM responses.

Instead of generating one answer, the system:

1. Generates an answer (Responder)
2. Critiques it (Critic)
3. Decides whether improvement is needed (Judge)
4. Iteratively refines the answer

It also includes:

- Short-Term Memory (STM): recent conversation
- Long-Term Memory (LTM): persistent vector memory (ChromaDB)

## How it works

```
User Input
   |
   v
Responder -> generates answer
   |
   v
Critic -> finds flaws
   |
   v
Judge -> decides if critique is useful
   |
   v
Improver -> refines answer (if needed)
   |
   v
Final Answer
```

Memory is injected into every step:

- STM: last few messages
- LTM: retrieved relevant past info

## Project Structure

```
.
├── main.py              # CLI application
├── requirements.txt     # dependencies
├── .env.example        # environment setup reference
├── experiment-notebooks/
│   ├── 01_llm_debate_system.ipynb
│   ├── 02_llm_debate_system.ipynb
│   └── 03_llm_debate_system.ipynb
└── README.md
```

## Running the project

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run

```bash
python main.py
```

3. Provide config

- API key (OpenRouter / NVIDIA / etc.)
- Base URL (default provided)
- Model (default provided)

Then start chatting:

```
> Your question here
```

## Why I built this

Most LLM apps generate a single response and stop there.

I wanted to explore:

- Can we simulate reasoning loops?
- Can multiple roles improve answer quality?
- How much improvement comes from structured critique?

This project is an attempt to understand that, not just use APIs blindly.

## What makes it interesting

- Multi-agent simulation without actual multiple models
- Explicit reasoning loop (not hidden chain-of-thought)
- Judge-based filtering (prevents over-optimization)
- Memory integration (STM + vector LTM)
- CLI-based, minimal, transparent system

It's simple enough to understand fully, but powerful enough to experiment with.

## Limitations

- Uses the same model for all roles (not truly multi-agent)
- No streaming / async (blocking calls)
- Memory is basic (no scoring, decay, or weighting)
- Prompt engineering dependent (can break easily)
- No evaluation metrics (quality is subjective)

## What I learned

- Iterative refinement works, but only if critique quality is good
- Judge layer is important to avoid unnecessary changes
- Memory injection must be controlled (too much = noise)
- Structured prompting > longer prompting
- Most complexity in LLM systems is orchestration, not generation

## Future improvements

- Different models per role
- Better memory ranking and filtering
- Tool usage (search, code execution)
- Streaming responses
- Web UI instead of CLI

## Final note

This is not a production system.

It's a learning-focused project to understand how reasoning systems can be built on top of LLMs.