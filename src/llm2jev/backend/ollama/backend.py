from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, asdict

import ollama

from ...core.response import Usage
from ...inference.prompt import ChatPrompt
from ..base import BinaryBackendOutput


@dataclass
class OllamaDiagnostic:
    prompt_index: int
    model: str

    generated_token: str

    yes_found: bool
    no_found: bool

    yes_rank: int | None
    no_rank: int | None

    yes_logprob: float | None
    no_logprob: float | None

    p_yes: float | None

    prompt_tokens: int
    output_tokens: int

    total_duration_ns: int | None

    @property
    def covered(self) -> bool:
        return self.yes_found and self.no_found

    def to_dict(self):
        return asdict(self) | {
            "covered": self.covered
        }


class OllamaBackend:
    """
    LLM2Jev binary scorer backed by Ollama.

    Ollama2Jev generates exactly one token and extracts
    'yes' and 'no' from Ollama's top-k first-token
    log-probability distribution.
    """

    def __init__(
        self,
        *,
        host: str = "http://localhost:11434",
        yes_label: str = "yes",
        no_label: str = "no",
        top_logprobs: int = 20,
        temperature: float = 0.0,
        think: bool = False,
        keep_alive: str = "5m",
    ) -> None:

        if not 1 <= top_logprobs <= 20:
            raise ValueError(
                "top_logprobs must be between 1 and 20"
            )

        self.client = ollama.Client(host=host)

        self.yes_label = yes_label
        self.no_label = no_label

        self.top_logprobs = top_logprobs
        self.temperature = temperature
        self.think = think
        self.keep_alive = keep_alive

        # Diagnostics from the most recent score() call.
        self.diagnostics: list[OllamaDiagnostic] = []


    def score(
        self,
        *,
        model: str,
        prompts: Sequence[ChatPrompt],
    ) -> BinaryBackendOutput:

        prompt_values = tuple(prompts)

        if not prompt_values:
            raise ValueError("prompts must not be empty")

        # Reset diagnostics for every score call.
        self.diagnostics = []

        yes_probabilities: list[float] = []

        input_tokens = 0
        output_tokens = 0

        for prompt_index, prompt in enumerate(prompt_values):

            messages = self._convert_prompt(prompt)

            response = self.client.chat(
                model=model,
                messages=messages,
                stream=False,
                think=self.think,

                options={
                    "temperature": self.temperature,
                    "num_predict": 1,
                },

                logprobs=True,
                top_logprobs=self.top_logprobs,
                keep_alive=self.keep_alive,
            )

            if not response.logprobs:
                raise RuntimeError(
                    f"Ollama returned no logprobs "
                    f"for model {model!r}"
                )

            first = response.logprobs[0]

            yes_rank, yes_logprob = self._find_token(
                first.top_logprobs,
                self.yes_label,
            )

            no_rank, no_logprob = self._find_token(
                first.top_logprobs,
                self.no_label,
            )

            yes_found = yes_logprob is not None
            no_found = no_logprob is not None

            prompt_token_count = int(
                response.prompt_eval_count or 0
            )

            output_token_count = int(
                response.eval_count or 0
            )

            # --------------------------------------------------
            # Coverage failure
            # --------------------------------------------------

            if not (yes_found and no_found):

                diagnostic = OllamaDiagnostic(
                    prompt_index=prompt_index,
                    model=model,

                    generated_token=first.token,

                    yes_found=yes_found,
                    no_found=no_found,

                    yes_rank=yes_rank,
                    no_rank=no_rank,

                    yes_logprob=yes_logprob,
                    no_logprob=no_logprob,

                    p_yes=None,

                    prompt_tokens=prompt_token_count,
                    output_tokens=output_token_count,

                    total_duration_ns=getattr(
                        response,
                        "total_duration",
                        None,
                    ),
                )

                self.diagnostics.append(diagnostic)

                available = [
                    item.token
                    for item in first.top_logprobs
                ]

                raise RuntimeError(
                    "Ollama2Jev top-logprob coverage failure. "
                    f"Model={model!r}, "
                    f"prompt_index={prompt_index}, "
                    f"yes_found={yes_found}, "
                    f"no_found={no_found}, "
                    f"generated={first.token!r}, "
                    f"available={available}"
                )

            # --------------------------------------------------
            # Compute binary probability
            # --------------------------------------------------

            p_yes = self._binary_probability(
                yes_logprob,
                no_logprob,
            )

            yes_probabilities.append(p_yes)

            diagnostic = OllamaDiagnostic(
                prompt_index=prompt_index,
                model=model,

                generated_token=first.token,

                yes_found=True,
                no_found=True,

                yes_rank=yes_rank,
                no_rank=no_rank,

                yes_logprob=yes_logprob,
                no_logprob=no_logprob,

                p_yes=p_yes,

                prompt_tokens=prompt_token_count,
                output_tokens=output_token_count,

                total_duration_ns=getattr(
                    response,
                    "total_duration",
                    None,
                ),
            )

            self.diagnostics.append(diagnostic)

            input_tokens += prompt_token_count
            output_tokens += output_token_count

        return BinaryBackendOutput(
            yes_probabilities=yes_probabilities,

            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )


    @staticmethod
    def _convert_prompt(
        prompt: ChatPrompt,
    ) -> list[dict[str, str]]:

        return [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in prompt
        ]


    @staticmethod
    def _find_token(
        top_logprobs,
        target: str,
    ) -> tuple[int | None, float | None]:

        for rank, item in enumerate(
            top_logprobs,
            start=1,
        ):

            if item.token == target:

                return (
                    rank,
                    float(item.logprob),
                )

        return None, None


    @staticmethod
    def _binary_probability(
        yes_logprob: float,
        no_logprob: float,
    ) -> float:

        maximum = max(
            yes_logprob,
            no_logprob,
        )

        yes_mass = math.exp(
            yes_logprob - maximum
        )

        no_mass = math.exp(
            no_logprob - maximum
        )

        return yes_mass / (
            yes_mass + no_mass
        )


    def get_diagnostics(self) -> list[dict]:
        """
        Return diagnostics for the most recent score() call.
        """
        return [
            item.to_dict()
            for item in self.diagnostics
        ]


    def coverage_rate(self) -> float:
        """
        Fraction of candidate evaluations where both exact
        lowercase 'yes' and 'no' were present in top-k.
        """

        if not self.diagnostics:
            return 0.0

        covered = sum(
            item.covered
            for item in self.diagnostics
        )

        return covered / len(self.diagnostics)

    def diagnostic_summary(self) -> dict:

        if not self.diagnostics:
            return {
                "candidate_evaluations": 0,
                "coverage_rate": 0.0,
            }

        covered = [
            item
            for item in self.diagnostics
            if item.covered
        ]

        yes_ranks = [
            item.yes_rank
            for item in covered
            if item.yes_rank is not None
        ]

        no_ranks = [
            item.no_rank
            for item in covered
            if item.no_rank is not None
        ]

        return {
            "candidate_evaluations":
                len(self.diagnostics),

            "covered":
                len(covered),

            "coverage_rate":
                len(covered)
                / len(self.diagnostics),

            "mean_yes_rank":
                sum(yes_ranks) / len(yes_ranks)
                if yes_ranks else None,

            "mean_no_rank":
                sum(no_ranks) / len(no_ranks)
                if no_ranks else None,

            "mean_p_yes":
                sum(
                    item.p_yes
                    for item in covered
                    if item.p_yes is not None
                ) / len(covered)
                if covered else None,

            "input_tokens":
                sum(
                    item.prompt_tokens
                    for item in self.diagnostics
                ),

            "output_tokens":
                sum(
                    item.output_tokens
                    for item in self.diagnostics
                ),

            "total_duration_ns":
                sum(
                    item.total_duration_ns or 0
                    for item in self.diagnostics
                ),
        }
