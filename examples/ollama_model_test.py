from llm2jev import (
    Choice,
    JevRequest,
    LLM2Jev,
    OllamaBackend,
)


MODELS = [
    "qwen3.5:0.8b",
    "llama3.2:1b",
    "llama3.2:3b",
    "granite4.1:3b",
    "gemma3:12b",
]


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


TEXT = "I feel absolutely wonderful today."


for model in MODELS:

    print()
    print("=" * 60)
    print("MODEL:", model)
    print("=" * 60)

    backend = OllamaBackend(
        top_logprobs=20,
        think=False,
        keep_alive="5m",
    )

    request = JevRequest(
        model=model,

        state=TEXT,

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

    try:

        response = LLM2Jev(
            backend=backend
        ).evaluate(request)

        answer = response.answers["emotion"]

        print(
            "Prediction:",
            answer.choice
        )

        print(
            "Confidence:",
            answer.confidence
        )

        print(
            "Probabilities:",
            answer.probabilities
        )

        print(
            "Diagnostics:",
            backend.diagnostic_summary()
        )

    except Exception as error:

        print(
            "FAILED:",
            type(error).__name__,
            error,
        )

        print(
            "Partial diagnostics:",
            backend.get_diagnostics()
        )
