from llm2jev import (
    Choice,
    JevRequest,
    LLM2Jev,
    OllamaBackend,
)


EMOTIONS = {
    "anger":
        "Angry, annoyed, irritated, or furious.",

    "fear":
        "Afraid, anxious, worried, or threatened.",

    "happy":
        "Happy, joyful, cheerful, or satisfied.",

    "love":
        "Love, affection, fondness, or strong liking.",

    "sadness":
        "Sad, disappointed, grieving, or hopeless.",
}


backend = OllamaBackend(
    top_logprobs=20,
    think=False,
)


request = JevRequest(
    model="qwen3.5:0.8b",

    state=(
        "I feel absolutely wonderful today."
    ),

    questions={
        "emotion": Choice(
            instructions=(
                "What single primary emotion "
                "is most dominant in this text?"
            ),

            criteria=EMOTIONS,
        )
    },
)


response = LLM2Jev(
    backend=backend
).evaluate(request)


print(response)

print("\n=== OLLAMA2JEV DIAGNOSTICS ===")

for row in backend.get_diagnostics():

    print(
        f"prompt={row['prompt_index']} "
        f"generated={row['generated_token']!r} "
        f"yes_rank={row['yes_rank']} "
        f"no_rank={row['no_rank']} "
        f"P(yes)={row['p_yes']:.4f}"
    )

print(
    "\nCoverage:",
    f"{backend.coverage_rate() * 100:.2f}%"
)
