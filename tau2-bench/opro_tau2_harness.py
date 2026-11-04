import json
import re
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import ollama
from datetime import datetime
from pathlib import Path
import shutil
from tau2 import TextRunConfig
from tau2.run import run_domain

import tau2.config as tau2_config

# config
AGENT_MODEL = "qwen2.5:7b"
OPTIMIZER_MODEL = "llama3.1:8b"
DOMAIN = "mock"  # replace with bespoke domain when its ready
NUM_OPTIMIZATION_STEPS = 5
CANDIDATES_PER_STEP = 2  # number of prompts Llama (Optimiser) will generate each cycle
MAX_HISTORY_IN_METAPROMPT = 8  # keep top/recent prompts so context doesn't overflow


@dataclass
class PromptRecord:
    step: int
    prompt: str
    score: float  # 0 to 100 (KPI / Pass rate)
    failure_summaries: List[str]




tau2_config.DEFAULT_LLM_NL_ASSERTIONS = "ollama_chat/qwen2.5:7b"
tau2_config.DEFAULT_LLM_ENV_INTERFACE = "ollama_chat/qwen2.5:7b"
tau2_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {
    "temperature": 0.0,
    "api_base": "http://127.0.0.1:11434",
}

POLICY_PATH = Path("data/tau2/domains/mock/policy.md")
BACKUP_PATH = Path("data/tau2/domains/mock/policy.md.bak")

def run_tau2_mock_domain(candidate_prompt: str, task_ids: list[str] | None = None) -> dict:

    # backup policy
    if not BACKUP_PATH.exists() and POLICY_PATH.exists():
        shutil.copy(POLICY_PATH, BACKUP_PATH)

    # safe savepath means no colons in directory name -> no windows errors!
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_save_path = Path(f"data/simulations/opro_run_{timestamp}/results.json")

    try:
        POLICY_PATH.write_text(candidate_prompt, encoding="utf-8")

        agent_ollama_args = {
            "temperature": 0.0,
            "api_base": "http://127.0.0.1:11434",
            "num_ctx": 4096,
        }
        user_ollama_args = {
            "temperature": 0.3,
            "api_base": "http://127.0.0.1:11434",
            "num_ctx": 4096,
        }

        config = TextRunConfig(
            domain="mock",
            agent="llm_agent",
            llm_agent="ollama_chat/qwen2.5:7b",
            llm_args_agent=agent_ollama_args,
            user="user_simulator",
            llm_user="ollama_chat/qwen2.5:7b",
            llm_args_user=user_ollama_args,
            task_ids=task_ids,
            num_trials=1,
            max_concurrency=1,
            save_to=str(safe_save_path),
        )

        simulation_results = run_domain(config)

        passed = 0
        failures = []
        simulations = simulation_results.simulations

        for sim in simulations:
            reward = sim.reward_info.reward if sim.reward_info else 0.0
            if reward == 1.0:
                passed += 1
            else:
                breakdown = sim.reward_info.model_dump() if sim.reward_info else {}
                failures.append(f"Task {sim.task_id} failed: {breakdown}")

        total = len(simulations) if simulations else 1
        score = round((passed / total) * 100.0, 2)

        return {
            "score": score,
            "failures": failures[:3]
        }

    finally:
        # restore original policy after being tampered with
        if BACKUP_PATH.exists():
            shutil.copy(BACKUP_PATH, POLICY_PATH)


def parse_tau2_results(raw_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Helper to convert standard tau2 trajectory JSON outputs into a single score (0-100)
    and concise failure descriptions for the OPRO meta-prompt.
    """
    total = len(raw_results)
    if total == 0:
        return {"score": 0.0, "failures": ["No tasks ran."]}

    passed = 0
    failures = []

    for res in raw_results:
        reward = res.get("reward", 0.0)
        if reward == 1.0:
            passed += 1
        else:
            task_id = res.get("task_id", "?")
            # get why it failed (e.g., database hash mismatch or wrong tool call)
            info = res.get("info", {})
            reason = info.get("reward_info", "Database final state did not match expected state.")
            failures.append(f"Task {task_id} failed: {reason}")

    score = round((passed / total) * 100.0, 2)
    # limit failure logs to top 3
    return {"score": score, "failures": failures[:3]}


def build_opro_meta_prompt(history: List[PromptRecord], domain_description: str) -> str:

    # sort history from lowest score to highest score
    sorted_history = sorted(history, key=lambda x: x.score)[-MAX_HISTORY_IN_METAPROMPT:]

    trajectory_str = ""
    for idx, item in enumerate(sorted_history, 1):
        fail_str = "\n    - ".join(item.failure_summaries) if item.failure_summaries else "None"
        trajectory_str += (
            f"\n--- Candidate #{idx} (Score: {item.score}/100) ---\n"
            f"Prompt:\n{item.prompt}\n"
            f"Observed Mistakes:\n    - {fail_str}\n"
        )

    meta_prompt = f"""You are an expert AI Prompt Optimizer using the OPRO (Optimization by Prompting) methodology.
        Your goal is to write a better system prompt for a customer-service & operations AI agent (Qwen 2.5 7B) operating in the '{DOMAIN}' domain on the tau2 benchmark.

        ### Domain & Operational Requirements:
        {domain_description}

        ### Previous Prompt Candidates and Their Evaluation Scores (Sorted Lowest to Highest):
        {trajectory_str}

        ### Optimization Instructions:
        1. Analyze why lower-scoring prompts failed (look at the "Observed Mistakes" above).
        2. Identify what higher-scoring prompts did right.
        3. Write a NEW system prompt that fixes the observed mistakes, enforces strict tool-use verification, and improves the evaluation score above {sorted_history[-1].score}.
        4. Keep the instructions clear, structured, and direct so a 7B parameter model can follow them reliably.
        5. Output ONLY the new system prompt wrapped strictly inside <PROMPT> and </PROMPT> tags.

        <PROMPT>
        """
    return meta_prompt


def generate_candidate_prompt(meta_prompt: str, temperature: float = 0.7) -> str:
    """
    Calls local Llama 3.1 8B via Ollama to generate a new candidate prompt.
    """
    response = ollama.chat(
        model=OPTIMIZER_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a precise meta-optimizer. Always wrap your final prompt suggestion inside <PROMPT> and </PROMPT> tags."
            },
            {
                "role": "user",
                "content": meta_prompt
            }
        ],
        options={
            "temperature": temperature,
            "num_predict": 600,
        }
    )

    raw_output = response["message"]["content"]

    # extract text between <PROMPT> and </PROMPT>
    match = re.search(r"<PROMPT>(.*?)</PROMPT>", raw_output, re.DOTALL)
    if match:
        return match.group(1).strip()

    if "</PROMPT>" in raw_output:
        return raw_output.split("</PROMPT>")[0].strip()

    return raw_output.strip()


def main():
    # define the mock domain rules
    domain_description = (
        "The agent assists users by looking up state in a database and calling tools to modify it. "
        "It must always verify constraints before mutating the database, never hallucinate tool outputs, "
        "and confirm policies before executing changes."
    )

    # initial seed prompt
    seed_prompt = "You are a helpful assistant. Use the provided tools to help the user with their request."

    train_task_ids = train_task_ids = [
        "create_task_1",
        "update_task_1",
    ]
    history: List[PromptRecord] = []

    print("=== Step 0: Evaluating Seed Prompt ===")
    eval_res = run_tau2_mock_domain(seed_prompt, train_task_ids)
    history.append(
        PromptRecord(
            step=0,
            prompt=seed_prompt,
            score=eval_res["score"],
            failure_summaries=eval_res["failures"]
        )
    )
    print(f"Seed Score: {eval_res['score']}/100\n")

    # run OPRO optimization loop
    for step in range(1, NUM_OPTIMIZATION_STEPS + 1):
        print(f"=== Optimization Step {step}/{NUM_OPTIMIZATION_STEPS} ===")
        meta_prompt = build_opro_meta_prompt(history, domain_description)

        for cand_idx in range(CANDIDATES_PER_STEP):
            print(f"  Generating candidate {cand_idx + 1} with {OPTIMIZER_MODEL}...")
            candidate_prompt = generate_candidate_prompt(meta_prompt, temperature=0.7)

            # evaluate candidate on tau2 with Qwen
            eval_res = run_tau2_mock_domain(candidate_prompt, train_task_ids)
            
            record = PromptRecord(
                step=step,
                prompt=candidate_prompt,
                score=eval_res["score"],
                failure_summaries=eval_res["failures"]
            )
            history.append(record)
            print(f"  Candidate {cand_idx + 1} Score: {record.score}/100")

    # save versioned prompt history to disk
    os.makedirs("optimization_logs", exist_ok=True)
    with open("optimization_logs/opro_history.json", "w") as f:
        json.dump([asdict(r) for r in history], f, indent=2)

    best = max(history, key=lambda x: x.score)
    print("\n==========================================")
    print(f"Best Prompt Found (Step {best.step} | Score: {best.score}/100):")
    print("==========================================")
    print(best.prompt)


if __name__ == "__main__":
    main()