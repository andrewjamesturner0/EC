# Pass 3: Final Editing Prompt

You are producing the final deliverable from the Empirical Cycling Podcast thematic analysis. All theme synthesis files have been completed. Your job is to edit them into a single, polished document.

## Instructions

Read all synthesis files below. Produce `output/final_output.md` with the following structure:

## Output Format

```markdown
# Empirical Cycling Podcast: Training Knowledge Base

## Executive Summary
[200-word overview of the most important cross-cutting findings and recommendations]

## Table of Contents
[Auto-generated from theme headings]

---

## 1. [Theme: Foundations / General Training Principles]
[Content from synthesis file, edited for consistency]

## 2. [Theme: Endurance / Zone 2]
...

## 3. [Theme: Threshold / FTP]
...

[Continue in logical order: Foundations → Endurance → Threshold → VO2max → Strength → Nutrition → Recovery → Racing → Metrics/Testing → Psychology/Coaching]

---

## Cross-References
[Key connections between themes, e.g. "Recovery recommendations interact with VO2max training frequency — see sections 4 and 7"]

## Methodology Note
[Brief note: 180 transcripts, ~3.1M words, clustered by TF-IDF similarity, extracted and synthesized in 3 passes. Not a replacement for listening to the podcast.]
```

## Editing Rules

1. **Cross-reference between themes.** Add "[See also: Section X]" links where themes interact.
2. **Remove cross-theme redundancy.** If the same recommendation appears in two themes, keep it in the most relevant one and cross-reference from the other.
3. **Consistent tone.** Professional, advisory, evidence-based. Not academic, not casual. Think "experienced coach summarizing the research."
4. **Consistent formatting.** Same heading levels, bullet styles, source citation format throughout.
5. **Logical ordering.** Arrange themes from foundational to specific: general principles → energy systems → training methods → supporting factors → performance.
6. **Preserve all source citations.** Every recommendation must still trace back to specific episodes.
7. **Executive summary should be genuinely useful** — the 5-10 most important takeaways that apply to most cyclists.

## Grounding Verification

After producing the final document, sample 5-10 specific factual claims. For each:
1. Identify the cited source episode
2. Read the relevant section of that transcript
3. Verify the claim accurately represents what was said
4. Flag any mismatches

Report verification results at the end of the document.

## Synthesis Files

{synthesis_files}
