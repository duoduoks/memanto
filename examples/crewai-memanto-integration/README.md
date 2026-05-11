# CrewAI + MEMANTO Integration

A production-ready drop-in memory provider for CrewAI agents using MEMANTO's persistent, queryable agent memory layer.

## Why This Matters

CrewAI agents default to **ephemeral** memory — context dies when the session ends.  
MEMANTO provides **cross-session, typed, queryable** memory with:

- 🧠 13 semantic memory types (fact, preference, goal, instruction, …)
- 📊 Confidence scoring + provenance metadata
- 🔄 Contradiction detection via versioning
- ⚡ Zero-ingestion-latency storage
- 🔍 Built-in semantic search + grounded QA

## Quick Start

```bash
pip install crewai memanto
export MOORCHEH_API_KEY="your-key-here"
```

## Usage

### As a drop-in memory provider

```python
from crewai import Agent
from memanto_memory import MemantoAgentMemory

agent = Agent(
    role="Researcher",
    goal="Find insights",
    memory=True,
    memory_provider=MemantoAgentMemory(
        agent_id="my_researcher",
        api_key="your-key",
    ),
)
```

### Cross-session demo

```bash
# Session 1: Research Agent stores findings
python crewai_memanto_demo.py --mode research

# Session 2: Writer Agent retrieves them (separate process!)
python crewai_memanto_demo.py --mode write
```

## Architecture

```
┌─────────────────────────────────────────────┐
│              CrewAI Agent                    │
│  ┌───────────────────────────────────────┐  │
│  │  MemantoAgentMemory                   │  │
│  │  ┌─────────────────────────────────┐  │  │
│  │  │  Memanto SDK (memanto package)   │  │  │
│  │  │  ┌───────────────────────────┐  │  │  │
│  │  │  │ moorcheh_sdk (storage)     │  │  │  │
│  │  │  └───────────────────────────┘  │  │  │
│  │  └─────────────────────────────────┘  │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Drop-in** | Implements CrewAI's expected memory interface |
| **Typed** | 13 semantic memory categories from MEMANTO |
| **Persistent** | Cross-session — survive restarts |
| **Queryable** | Semantic search, not just injected context |
| **Confidence** | Every memory has a confidence score |
| **Contradictions** | Versioning detects and marks conflicts |
| **Grounded QA** | Built-in RAG via MemantoClient.answer() |

## License

MIT
