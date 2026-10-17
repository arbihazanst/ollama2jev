# Z2–Z4 Offline Ablation — Atomic Manuscript Revisions

**Target:** `access(8).tex`, the uploaded IEEE Access manuscript. **Data:** 11 model-specific `ollama2jev_jevbench_*_full.jsonl` files at `arbihazanst/ollama2jev`, branch `ollama2jev`, plus JevBench `original.jsonl`, `easy.jsonl`, `hard.jsonl` (231 unique tasks).

**Status:** The Z2, Z3, and Z4 results below were **computed from saved first-token log-probabilities**, without contacting Ollama or rerunning models. The Z1 predictions were independently reconstructed and matched the logged `correct` flags for all 2,538 successful decisions. Three Granite 4.2 8B decisions failed at source and remain failures in every ablation. **Z0 (direct zero-shot generation) has not been run** and is not reported as evidence.

## Reproduction and operational definitions

`Z1`: existing complete Ollama2Jev binary score $q_i=\exp(\ell_{y,i})/[\exp(\ell_{y,i})+\exp(\ell_{n,i})]$, followed by the unchanged LLM2Jev typed assembler.

`Z2`: replace each candidate score with $s_i=\exp(\ell_{y,i})$, then retain the same L1 normalization for multi-candidate Choice/Score and criteria-based Noul, and the original threshold for criterion-free Noul. This is the **unconditional yes-likelihood readout**, not an ablation of the final L1 assembler.

`Z3`: use $h_i=\mathbb{I}[q_i\geq 0.5]$ for each candidate, then apply the same assembler, including zero-sum uniform fallback and first-listed-candidate tie-breaking. This intentionally coarse quantizer creates ties and can assign exact zero to the gold class.

`Z4`: for $k\in\{2,5,10,20\}$, use first-token `yes_rank` and `no_rank` already saved under top-20. A candidate is covered only when **both** ranks are $\leq k$; a typed decision is covered only when *all recorded binary candidates* meet the condition and the original typed response was successful. This is an **offline top-k truncation replay**, *not* a fresh model run with `top_logprobs=k`. Attempts/candidate denominators correspond to the existing full-run log, including early termination on the three Granite failures.

Primary accuracy denominator: 231 tasks per model or 2,541 model–task attempts pooled; uncovered/failed items count as incorrect. Brier uses only complete typed distributions (2,538 decisions) and the existing unnormalized multiclass Brier definition. `Z3` NLL should **not** be used as evidence of calibrated probabilities because deterministic zero probabilities make it clipping-dependent. ECE is omitted from the main ablation tables because hard-score ties and model-specific binning can make comparisons misleading; the reproducibility script exports ECE values for supplementary inspection.

Run the supplied `reproduce_z2_z4_ablation.py` from the repository root after copying it there:

```bash
python reproduce_z2_z4_ablation.py --results-dir . --dataset-dir ../jevbench/datasets/public --out-dir ablation_offline
```

---

## Atomic edit 1 — Research question (Introduction, ~line 137)
**Find** the sentence: `The paper is organized around four research questions.` **Replace the whole research-question paragraph** (through `...new model architecture.`) with:
```latex
The paper is organized around five research questions. \textbf{RQ1} addresses the feasibility of typed one-token decisions with local LLMs. \textbf{RQ2} examines model-family and size variation. \textbf{RQ3} analyzes Choice, Noul, and Score sensitivity. \textbf{RQ4} examines coverage and latency. \textbf{RQ5} tests how much the observed decisions depend on binary yes/no renormalization, retention of soft candidate scores, and the maximum number of returned alternatives. The work is a systems-and-evaluation contribution rather than a new model architecture.
```

## Atomic edit 2 — Contribution bullet (~lines 147–153)
**Find:** `The main contributions of this work are threefold:` **Replace only that sentence** with:
```latex
The main contributions of this work are fourfold:
```

## Atomic edit 3 — Add one contribution bullet
**Find** the last `\item` ending with `probability behavior, and failure modes.` **Immediately after that item, before `\end{itemize}`, add:**
```latex
\item We conduct reproducible offline scoring ablations from the recorded candidate log-probabilities, comparing the complete readout with yes-only likelihood scoring, hard candidate scores, and effective top-$k$ truncation at $k\in\{2,5,10,20\}$. The ablations identify which implementation choices affect decisions without confounding them with new model calls.
```

## Atomic edit 4 — Research question under Experimental Methodology (~line 410)
**Find** `\textbf{RQ4---Practicality:}` and its complete question. **After that paragraph, before** `The study is exploratory and systems-oriented.`, insert:
```latex
\textbf{RQ5---Scoring sensitivity:} How do binary yes/no renormalization, hard thresholding of candidate probabilities, and top-$k$ truncation influence decision accuracy, probability quality, and protocol coverage?
```

## Atomic edit 5 — Add the experimental design (~line 525)
**Find** the end of the paragraph starting `The benchmark evaluated 11 Ollama model configurations`. **Insert immediately before** `\subsection{Metrics}`:
```latex
\subsection{Offline Scoring Ablation Design}
\label{subsec:offline-ablation}
We evaluate three offline variants of the saved first-token candidate readouts using the same 11 model configurations and 231 public JevBench items. The complete Ollama2Jev method (Z1) uses the binary-conditional yes probability
\begin{equation}
q_i=\frac{\exp(\ell_{y,i})}{\exp(\ell_{y,i})+\exp(\ell_{n,i})},
\end{equation}
where $\ell_{y,i}$ and $\ell_{n,i}$ are the recorded first-token log-probabilities of the exact lowercase \texttt{yes} and \texttt{no} tokens for candidate $i$.

Z2 removes the binary renormalization step and uses the unconditional yes-token likelihood
\begin{equation}
s_i^{\mathrm{Z2}}=\exp(\ell_{y,i}).
\end{equation}
Z3 replaces the continuous binary-conditional score with a hard score,
\begin{equation}
s_i^{\mathrm{Z3}}=\mathbb{I}[q_i\geq 0.5].
\end{equation}
For each scoring variant, the original LLM2Jev Choice, Noul, and Score response assembly is preserved, including L1 normalization for multi-candidate questions, uniform fallback for zero-sum vectors, the default first-candidate tie break, and the 0.5 threshold for scalar Noul.

Z4 investigates effective first-token coverage thresholds $k\in\{2,5,10,20\}$ by filtering the recorded top-20 diagnostics. A candidate is considered available at $k$ only if both exact decision-token ranks are at most $k$. A typed decision is counted as successful only when all its candidates are available. Its label is then the original Z1 prediction, because lower-$k$ truncation does not change the two recorded log-probabilities when both tokens remain visible. The calculation replays saved diagnostic ranks; it does not measure execution time under a new Ollama $k$ setting.

All variants are analyzed on exactly the same stored decision attempts. Uncovered or incomplete typed decisions count as incorrect for overall accuracy. Probability metrics are computed only when a complete typed-decision distribution can be reconstructed. These offline ablations cannot replace a separately executed zero-shot generative baseline.
```

## Atomic edit 6 — Extend the Metrics subsection (~line 568)
**Find** `\subsubsection{Top-$k$ Coverage}`. **After its existing paragraph**, before `\subsubsection{Yes Bias and Decision Margin}`, insert:
```latex
\subsubsection{Ablation Outcome Measures}
For Z1--Z3, we compute overall exact-match accuracy using the full 231-task denominator per model, while Brier score uses successful typed distributions only. We also record paired correctness changes relative to Z1. For Z4, we report candidate-level coverage over recorded binary calls, typed-decision completion over all 2,541 model--task attempts, and strict full-denominator accuracy that counts coverage failures as incorrect. We do not impute missing logits or probabilities. Hard-thresholded probabilities in Z3 are analyzed primarily for accuracy and Brier score; their exact-zero class masses make NLL highly sensitive to numerical clipping.
```

## Atomic edit 7 — Insert Z1–Z3 result table (~line 818)
**Find** `\subsection{Failure Analysis}` in Results. **Immediately before it, insert this entire table** (then add the following edits in order):
```latex
\begin{table*}[!t]
\caption{Offline Scoring Ablation Accuracy on the 231 Public JevBench Tasks. Z1 is the complete Ollama2Jev binary-conditional score; Z2 uses unconditional yes-token likelihood; Z3 hard-thresholds each binary score before unchanged typed assembly. All accuracies use the 231-task denominator; incomplete responses count as incorrect.}
\label{tab:offline-ablation-accuracy}
\centering
\small
\begin{tabular}{lrrr}
\toprule
\textbf{Model} & \textbf{Z1 (full)} & \textbf{Z2 (yes-only)} & \textbf{Z3 (hard)} \\
\midrule
Qwen3.5 0.8B & 0.519 (120/231) & 0.519 (120/231) & 0.368 (85/231) \\
Qwen3.5 2B & 0.641 (148/231) & 0.636 (147/231) & 0.563 (130/231) \\
Qwen3.5 4B & 0.710 (164/231) & 0.710 (164/231) & 0.697 (161/231) \\
Qwen3.5 9B & 0.723 (167/231) & 0.723 (167/231) & 0.693 (160/231) \\
Qwen3.8 27B & 0.853 (197/231) & 0.853 (197/231) & 0.654 (151/231) \\
Llama 3.2 1B & 0.398 (92/231) & 0.394 (91/231) & 0.303 (70/231) \\
Llama 3.2 3B & 0.667 (154/231) & 0.671 (155/231) & 0.489 (113/231) \\
Granite 4.2 3B & 0.649 (150/231) & 0.641 (148/231) & 0.571 (132/231) \\
Granite 4.2 8B & 0.636 (147/231) & 0.636 (147/231) & 0.554 (128/231) \\
Gemma 4 E2B & 0.623 (144/231) & 0.615 (142/231) & 0.498 (115/231) \\
Gemma 4 E4B & 0.714 (165/231) & 0.714 (165/231) & 0.619 (143/231) \\
\midrule
\textbf{Pooled} & 0.649 (1648/2541) & 0.647 (1643/2541) & 0.546 (1388/2541) \\
\bottomrule
\end{tabular}
\end{table*}
```

## Atomic edit 8 — Insert Z1–Z3 Brier table
**Immediately after** the new `tab:offline-ablation-accuracy` table, insert:
```latex
\begin{table*}[!t]
\caption{Offline Scoring Ablation Probability Quality: Multiclass Brier Score. Lower is better. Metrics use only complete typed distributions (231 per model except Granite 4.2 8B, with 228). Hard Z3 distributions are discrete and often degenerate; a lower Brier score for one model does not establish general calibration superiority.}
\label{tab:offline-ablation-brier}
\centering
\small
\begin{tabular}{lrrr}
\toprule
\textbf{Model} & \textbf{Z1 (full)} & \textbf{Z2 (yes-only)} & \textbf{Z3 (hard)} \\
\midrule
Qwen3.5 0.8B & 0.647 & 0.647 & 0.734 \\
Qwen3.5 2B & 0.448 & 0.451 & 0.567 \\
Qwen3.5 4B & 0.404 & 0.404 & 0.401 \\
Qwen3.5 9B & 0.360 & 0.360 & 0.437 \\
Qwen3.8 27B & 0.193 & 0.193 & 0.314 \\
Llama 3.2 1B & 0.677 & 0.696 & 0.743 \\
Llama 3.2 3B & 0.494 & 0.493 & 0.633 \\
Granite 4.2 3B & 0.464 & 0.473 & 0.540 \\
Granite 4.2 8B & 0.557 & 0.557 & 0.496 \\
Gemma 4 E2B & 0.611 & 0.607 & 0.546 \\
Gemma 4 E4B & 0.417 & 0.417 & 0.465 \\
\midrule
\textbf{Pooled (2538)} & 0.479 & 0.482 & 0.534 \\
\bottomrule
\end{tabular}
\end{table*}
```

## Atomic edit 9 — Insert per-model effective top-k coverage table
**Immediately after** the new `tab:offline-ablation-brier` table, insert:
```latex
\begin{table*}[!t]
\caption{Offline Effective Top-$k$ Replay: Completed Typed Decisions out of 231 per Model. A decision is counted as covered only if every recorded binary candidate contains both exact decision tokens within its first $k$ ranked alternatives. These are replayed coverage counts, not new inference runs.}
\label{tab:offline-ablation-topk-model}
\centering
\small
\begin{tabular}{lrrrr}
\toprule
\textbf{Model} & $k=2$ & $k=5$ & $k=10$ & $k=20$ \\
\midrule
Qwen3.5 0.8B & 231/231 & 231/231 & 231/231 & 231/231 \\
Qwen3.5 2B & 224/231 & 231/231 & 231/231 & 231/231 \\
Qwen3.5 4B & 231/231 & 231/231 & 231/231 & 231/231 \\
Qwen3.5 9B & 228/231 & 231/231 & 231/231 & 231/231 \\
Qwen3.8 27B & 60/231 & 205/231 & 231/231 & 231/231 \\
Llama 3.2 1B & 201/231 & 223/231 & 229/231 & 231/231 \\
Llama 3.2 3B & 212/231 & 231/231 & 231/231 & 231/231 \\
Granite 4.2 3B & 131/231 & 230/231 & 231/231 & 231/231 \\
Granite 4.2 8B & 123/231 & 166/231 & 211/231 & 228/231 \\
Gemma 4 E2B & 120/231 & 212/231 & 230/231 & 231/231 \\
Gemma 4 E4B & 224/231 & 230/231 & 231/231 & 231/231 \\
\midrule
\textbf{Pooled} & 1985/2541 & 2421/2541 & 2518/2541 & 2538/2541 \\
\bottomrule
\end{tabular}
\end{table*}
```

## Atomic edit 10 — Insert aggregate effective top-k trade-off table
**Immediately after** the new `tab:offline-ablation-topk-model` table, insert:
```latex
\begin{table*}[!t]
\caption{Aggregate Offline Top-$k$ Sensitivity for the 11-Model JevBench Benchmark. Binary candidate coverage uses 9,327 recorded attempted calls, while typed completion and strict accuracy use 2,541 requested decisions. Uncovered typed decisions are scored incorrect.}
\label{tab:offline-ablation-topk-summary}
\centering
\small
\begin{tabular}{rrrrr}
\toprule
$k$ & \textbf{Binary coverage} & \textbf{Typed completed} & \textbf{Strict correct} & \textbf{Accuracy} \\
\midrule
2 & 7989/9327 & 1985/2541 & 1208/2541 & 0.475 \\
5 & 9106/9327 & 2421/2541 & 1552/2541 & 0.611 \\
10 & 9292/9327 & 2518/2541 & 1632/2541 & 0.642 \\
20 & 9324/9327 & 2538/2541 & 1648/2541 & 0.649 \\
\bottomrule
\end{tabular}
\end{table*}
```

## Atomic edit 11 — Insert Z2/Z3 findings in Results
**Immediately after** the `tab:offline-ablation-topk-summary` table, insert this new subsection and paragraph:
```latex
\subsection{Offline Scoring Ablation Results}
\label{subsec:offline-ablation-results}
Tables~\ref{tab:offline-ablation-accuracy} and~\ref{tab:offline-ablation-brier} compare the full binary-conditional readout (Z1) against unconditional yes-token scoring (Z2) and hard candidate scoring (Z3). Z1 reproduces all 11 recorded full-pipeline accuracy counts exactly. Removing binary yes/no renormalization in Z2 changes the pooled number of correct predictions only from 1,648 to 1,643 out of 2,541 model--task attempts (64.9\% versus 64.7\%). Its pooled Brier score changes from 0.479 to 0.482 on the 2,538 completed responses. The effect is small and inconsistent across models: Qwen3.8 27B remains at 197/231 under both approaches, whereas Llama 3.2 1B changes from 92 to 91 correct responses and Granite 4.2 3B from 150 to 148. Therefore, the data do not support a general accuracy advantage for binary renormalization on this public benchmark.

The hard-score variant Z3 has a larger effect. It reduces pooled correct decisions from 1,648 to 1,388 (64.9\% to 54.6\%) and increases pooled Brier score from 0.479 to 0.534. The strongest model, Qwen3.8 27B, drops from 197/231 (85.3\%) to 151/231 (65.4\%). The result suggests that retaining graded candidate probabilities is important for the typed assembler, especially when hard thresholds produce ties and zero-sum candidate vectors. However, the effect is not uniform: for Qwen3.5 4B, accuracy changes only from 164/231 to 161/231. These offline manipulations do not assess generative zero-shot prompting, which requires a separate inference experiment.
```

## Atomic edit 12 — Insert Z4 findings in Results
**Immediately after** the preceding two paragraphs, add:
```latex
The effective top-$k$ replay in Tables~\ref{tab:offline-ablation-topk-model} and~\ref{tab:offline-ablation-topk-summary} reveals a strong reliability--truncation trade-off. At $k=2$, only 1,985/2,541 typed decisions (78.1\%) are fully covered, yielding strict overall accuracy of 47.5\% (1,208/2,541). Increasing the threshold to $k=5$ raises typed completion to 2,421/2,541 (95.3\%) and strict accuracy to 61.1\%; at $k=10$, the corresponding values are 2,518/2,541 (99.1\%) and 64.2\%. The recorded top-20 condition covers 2,538/2,541 typed decisions (99.88\%) and retains the original 64.9\% strict accuracy. Candidate coverage increases from 7,989/9,327 (85.65\%) at $k=2$ to 9,324/9,327 (99.97\%) at $k=20$.

Top-$k$ sensitivity is model dependent. Qwen3.8 27B has only 60/231 fully covered decisions at $k=2$ and 205/231 at $k=5$, despite 231/231 coverage at $k=10$ and $k=20$. In contrast, Qwen3.5 0.8B and 4B retain 231/231 coverage already at $k=2$. This indicates that choosing a small top-$k$ on the basis of one model's token-rank distribution can substantially understate availability for another model. Since these values were derived from previously recorded top-20 token ranks, they do not establish the latency impact of requesting a smaller $k$ in a new Ollama run.
```

## Atomic edit 13 — Add a Discussion subsection (~line 856)
**Find** `\subsection{Is Top-20 Enough?}` in Discussion. **Immediately before it, insert:**
```latex
\subsection{What the Offline Ablations Establish}
\label{subsec:offline-ablation-discussion}
The ablations separate two design questions that would otherwise be conflated. First, unconditional yes-token likelihood and binary-conditional yes/no likelihood are empirically very similar under the current prompt and models; Z2 preserves nearly all of Z1's accuracy. Consequently, the data do not justify describing binary renormalization as the primary source of accuracy gains. Second, the conversion of continuous candidate scores into hard binary indicators causes a substantial loss of information for several models. The upstream LLM2Jev typed assembler relies on comparisons among candidate magnitudes, not simply on independent binary accept/reject labels. Third, top-$k$ truncation controls whether those candidate magnitudes are available at all, and its impact is highly model dependent.

These findings characterize the operating assumptions of an accessible local inference backend. They do not demonstrate superiority over direct zero-shot generation or exact-logit scoring, which remain separate baseline and fidelity questions. The ablations were performed offline on fixed model outputs and therefore do not address inference-time prompt variation, random decoding, or resource consumption under different top-$k$ API settings.
```

## Atomic edit 14 — Add a Threat to Validity (~line 882)
**Find** `\subsection{Verbalizer Dependence}` in Threats to Validity. **Immediately before it, insert:**
```latex
\subsection{Offline Replay Versus Prospective Inference}
The scoring ablations reconstruct counterfactual typed decisions from previously saved token log-probabilities rather than repeating inference under new serving configurations. Z2 and Z3 change only post-processing of the available scores. Z4 removes decisions whose required verbalizers would not have appeared within a smaller recorded rank cutoff; it does not issue requests with a smaller \code{top\_logprobs} setting. Therefore, the ablation outcomes quantify dependence on observed scores and coverage but do not establish counterfactual latency, memory use, or sampling behavior. Strict accuracy assigns uncovered decisions an incorrect outcome, while probability-quality metrics exclude incomplete distributions. Hard-thresholded candidate scores also introduce deterministic ties, so the results depend on the documented default tie-breaking and zero-sum normalization behavior. Moreover, pooled 11-model counts reuse the same 231 tasks and should be treated as descriptive rather than as 2,541 statistically independent samples.
```

## Atomic edit 15 — Update the Reproducibility section (~line 908)
**Find** the paragraph beginning `Each successful benchmark record contains`. **Immediately after that paragraph, insert:**
```latex
The offline Z2--Z4 ablations can be reconstructed from the saved per-candidate \code{yes\_logprob}, \code{no\_logprob}, \code{p\_yes}, \code{yes\_rank}, and \code{no\_rank} fields, together with the original JevBench task specifications. The accompanying replay script checks that the reconstructed Z1 predictions match the logged full-pipeline outcomes before computing ablation tables. No additional model generation is required for these particular scoring and coverage variants.
```

**Repository action:** add `reproduce_z2_z4_ablation.py` and its output CSVs to GitHub before asserting in the paper that the *ablation* artifacts are publicly available. Until then, describe them as accompanying analysis artifacts, not as already published.

## Atomic edit 16 — Update the Conclusion (~line 920)
**Find** the paragraph beginning `Ollama2Jev should be viewed as an accessibility and benchmarking contribution`. **Insert the following new paragraph immediately before it:**
```latex
The offline ablation study further clarifies the contribution of the scoring procedure. Removing binary yes/no renormalization has little net effect on decision accuracy (1,648 versus 1,643 correct model--task outcomes), whereas hard-thresholding candidate scores reduces the number of correct outcomes to 1,388. Effective top-$k$ replay shows that full typed-decision coverage rises from 78.1\% at $k=2$ to 99.88\% at $k=20$, with substantial variation across model families. These findings support retaining graded candidate scores and explicit coverage diagnostics, but do not substitute for a future direct-generation baseline or an exact-logit backend fidelity comparison.
```

## Atomic edit 17 — Optional one-sentence abstract update (~line 86)
**Find:** `The results demonstrate the feasibility of one-token top-logprob inference` at the end of the Abstract. **Immediately before that sentence, insert:**
```latex
Offline scoring ablations show little accuracy change when replacing binary-conditional scoring with unconditional yes-token likelihood (1,648 versus 1,643 correct model--task outcomes), but a marked decrease when candidate probabilities are hard-thresholded (1,388 correct); top-$k$ replay reveals model-dependent token-coverage sensitivity.
```

**Abstract length note:** The existing abstract is already long. If the journal limit is binding, shorten the preceding model-by-model performance sentences rather than appending this sentence without reduction.

---

## Validation checklist

- [ ] Z1 logged correctness reproduced **exactly**: 2,538 completed decisions, 1,648 correct; three pre-existing Granite 8B failures.
- [ ] Z2: 1,643/2,541 correct; Brier 0.482 (completed distributions only).
- [ ] Z3: 1,388/2,541 correct; Brier 0.534 (completed distributions only).
- [ ] Z4 k=2/5/10/20 typed coverages: 1,985 / 2,421 / 2,518 / 2,538 of 2,541.
- [ ] All new labels unique: `subsec:offline-ablation`, `tab:offline-ablation-accuracy`, `tab:offline-ablation-brier`, `tab:offline-ablation-topk-model`, `tab:offline-ablation-topk-summary`, `subsec:offline-ablation-results`, `subsec:offline-ablation-discussion`.
- [ ] No claim that Z0 has been executed; no claim that Z4 measures new Ollama latency.
- [ ] Compare from the GitHub file commit used for the analysis, not an unpinned mutable later copy; archive the model tags and revision IDs.

## Interpretation caveat relevant to the desk rejection

These are real **post-hoc computational experiments** on recorded inference distributions, not new model-serving experiments. They improve method characterization and transparency, but alone do **not** close the editor's request for direct comparisons against established methods. A prospective direct zero-shot generation baseline (Z0), constrained-generation reference, and ideally a matched exact-logit backend comparison remain recommended.
