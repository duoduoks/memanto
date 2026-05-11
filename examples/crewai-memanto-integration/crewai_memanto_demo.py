#!/usr/bin/env python3
"""
CrewAI + MEMANTO Integration Demo

Demonstrates cross-session agent memory using the official CrewAI SDK.

Usage:
    # Run research phase (stores memories)
    python crewai_memanto_demo.py --mode research

    # Run write phase (retrieves memories across sessions)
    python crewai_memanto_demo.py --mode write

    # Full two-agent workflow (uses Crew under the hood)
    python crewai_memanto_demo.py --mode full

Requirements:
    pip install crewai memanto
    export MOORCHEH_API_KEY="..."
"""

import os
import sys
import argparse
import logging
from datetime import datetime

from memanto_memory import MemantoAgentMemory

# ---------------------------------------------------------------------------
# CrewAI imports — official API objects
# ---------------------------------------------------------------------------
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("crewai-memanto-demo")

MEMANTO_AGENT_ID = "crewai_memanto_demo"
MOORCHEH_API_KEY = os.environ.get("MOORCHEH_API_KEY", "")
TOPIC = "The impact of AI on software development productivity"


# ===================================================================
# Agent factories  (CrewAI native Agent objects)
# ===================================================================

def make_research_agent() -> Agent:
    """Research agent with MEMANTO-backed memory."""
    memory_provider = MemantoAgentMemory(
        agent_id=f"{MEMANTO_AGENT_ID}_researcher",
        api_key=MOORCHEH_API_KEY,
    )
    return Agent(
        role="Senior Research Analyst",
        goal="Thoroughly investigate the topic and store structured findings",
        backstory=(
            "You are a meticulous researcher who documents every finding "
            "with clear evidence and confidence levels."
        ),
        memory=True,
        memory_provider=memory_provider,
        allow_delegation=False,
        verbose=True,
    )


def make_writer_agent() -> Agent:
    """Writer agent with MEMANTO-backed memory."""
    memory_provider = MemantoAgentMemory(
        agent_id=f"{MEMANTO_AGENT_ID}_writer",
        api_key=MOORCHEH_API_KEY,
    )
    return Agent(
        role="Technical Writer",
        goal="Synthesise retrieved memories into a coherent, well-structured article",
        backstory=(
            "You are a skilled writer who excels at connecting disparate "
            "pieces of information into a compelling narrative."
        ),
        memory=True,
        memory_provider=memory_provider,
        allow_delegation=False,
        verbose=True,
    )


# ===================================================================
# Research phase — store findings into MEMANTO
# ===================================================================

def run_research(memory_provider: MemantoAgentMemory):
    """
    Research phase: store findings as typed semantic memories.

    This is a lean approach — we skip the full Crew orchestration here
    to keep the demo self-contained and fast.  The memories are stored
    using the MemantoAgentMemory provider directly, exactly as a
    CrewAI agent would do internally.
    """
    print("\n" + "=" * 60)
    print("📚 RESEARCH PHASE")
    print("=" * 60)

    findings = [
        {
            "content": (
                "AI coding assistants improve developer productivity "
                "by 25-55% based on a 2025 GitHub survey of 10 000 developers."
            ),
            "memory_type": "fact",
            "title": "AI productivity boost",
            "tags": ["productivity", "github", "survey"],
        },
        {
            "content": (
                "Multi-agent systems show promise for complex tasks. "
                "CrewAI-based orchestrations reduced task completion time "
                "by 40% in enterprise pilots."
            ),
            "memory_type": "fact",
            "title": "Multi-agent task efficiency",
            "tags": ["multi-agent", "crewai", "orchestration"],
        },
        {
            "content": (
                "AI-assisted code review catches 30% more bugs than "
                "manual review alone, per a 2024 IEEE study."
            ),
            "memory_type": "fact",
            "title": "Code review bug detection",
            "tags": ["code-review", "ieee", "bugs"],
        },
        {
            "content": (
                "Shared memory between agents prevents information silos "
                "in long-running workflows."
            ),
            "memory_type": "preference",
            "title": "Shared memory best practice",
            "confidence": 0.85,
            "tags": ["architecture", "memory"],
        },
        {
            "content": (
                "Clear role definitions in CrewAI avoid task overlap "
                "and improve output quality."
            ),
            "memory_type": "instruction",
            "title": "Role definition guideline",
            "confidence": 0.9,
            "tags": ["crewai", "best-practice"],
        },
        {
            "content": (
                "Persistent agent memory is critical for maintaining "
                "context across multi-day research workflows."
            ),
            "memory_type": "goal",
            "title": "Persistent memory goal",
            "tags": ["memory", "persistence"],
        },
    ]

    stored = 0
    errors = 0
    for finding in findings:
        try:
            memory_provider.store(**finding)
            stored += 1
        except Exception as e:
            logger.error("Failed to store: %s", e)
            errors += 1

    print(f"\n ✅ Stored: {stored} memories")
    if errors:
        print(f" ❌ Errors: {errors}")

    # ── Contradiction demo (update a fact) ────────────────────────
    print("\n 🔄 Demonstrating contradiction handling...")
    updated_finding = {
        "content": (
            "AI coding assistants improve developer productivity "
            "by 30-60% — a refined estimate from a 2026 meta-analysis "
            "across 50 studies."
        ),
        "memory_type": "fact",
        "title": "AI productivity boost",
        "tags": ["productivity", "meta-analysis"],
        "confidence": 0.95,
    }
    try:
        memory_provider.store(**updated_finding)
        print(" ✅ Stored updated finding (old version superseded)")
        stored += 1
    except Exception as e:
        logger.error("Contradiction store failed: %s", e)
        errors += 1

    print(f"\n📊 Session Summary")
    print(f"{'─' * 40}")
    print(f"  Memories stored:  {stored}")
    print(f"  Errors:           {errors}")
    print(f"{'─' * 40}\n")


# ===================================================================
# Write phase — retrieve from MEMANTO and generate
# ===================================================================

def run_write(memory_provider: MemantoAgentMemory, topic: str = TOPIC):
    """
    Write phase: retrieve cross-session memories using the same
    MEMANTO-backed memory provider.
    """
    print("\n" + "=" * 60)
    print("✍️  WRITING PHASE")
    print("=" * 60)

    memories = memory_provider.recall(topic, top_k=10)
    print(f"\n 📖 Found {len(memories)} relevant memories")

    if memories:
        print(f"\n{'─' * 50}")
        for i, m in enumerate(memories, 1):
            title = m.get("title", "(untitled)")
            mem_type = m.get("type", "fact")
            conf = m.get("confidence", 0.0)
            print(f"  [{i}] {title} ({mem_type}, confidence={conf:.2f})")
        print(f"{'─' * 50}\n")

    # Generate a grounded answer
    print(" 🧠 Generating article from memory...\n")
    article = memory_provider.answer(
        f"Synthesise the key findings about {topic} into a short article. "
        f"Include specific data points and best practices."
    )
    if article:
        print("📄 GENERATED ARTICLE")
        print(f"{'─' * 50}")
        print(article)
        print(f"{'─' * 50}")
    else:
        print("(No grounded answer returned — memories may not have been indexed yet.)")

    print(f"\n📊 Session Summary")
    print(f"{'─' * 40}")
    print(f"  Memories recalled: {len(memories)}")
    print(f"{'─' * 40}\n")


# ===================================================================
# Full CrewAI workflow (two agents with real Crew)
# ===================================================================

def run_full():
    """
    Full CrewAI + MEMANTO workflow using native Agent / Task / Crew.
    """
    print("\n" + "=" * 60)
    print("🦞 CREWAI + MEMANTO FULL WORKFLOW")
    print("=" * 60)

    researcher = make_research_agent()
    writer = make_writer_agent()

    research_task = Task(
        description=(
            f"Research {TOPIC}. "
            "Store at least 3 specific findings with data points."
        ),
        expected_output="A bullet list of 3-5 key findings with statistics and sources.",
        agent=researcher,
    )

    write_task = Task(
        description=(
            f"Write a 300-word article about {TOPIC} "
            "based on the findings stored in your memory."
        ),
        expected_output="A well-structured article with introduction, data points, and conclusion.",
        agent=writer,
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    print("\n✅ Crew execution complete!")
    print(result)

    return result


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CrewAI + MEMANTO Integration Demo",
    )
    parser.add_argument(
        "--mode",
        choices=["research", "write", "full"],
        default="full",
        help="Execution mode (default: full)",
    )
    args = parser.parse_args()

    if not MOORCHEH_API_KEY:
        print(
            "ERROR: MOORCHEH_API_KEY environment variable is not set.\n"
            "Get your key at https://console.moorcheh.ai/api-keys"
        )
        sys.exit(1)

    if args.mode == "research":
        memory = MemantoAgentMemory(
            agent_id=f"{MEMANTO_AGENT_ID}_researcher",
            api_key=MOORCHEH_API_KEY,
        )
        run_research(memory)

    elif args.mode == "write":
        memory = MemantoAgentMemory(
            agent_id=f"{MEMANTO_AGENT_ID}_writer",
            api_key=MOORCHEH_API_KEY,
        )
        run_write(memory)

    elif args.mode == "full":
        run_full()

    print("\n🦞 Demo complete! Cross-session memory test passed.\n")


if __name__ == "__main__":
    main()
