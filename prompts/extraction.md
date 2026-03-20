# Pass 1: Extraction Prompt

You are extracting actionable training advice from Empirical Cycling Podcast transcripts. These transcripts belong to the **"{cluster_name}"** topic cluster.

## Instructions

Read the transcript(s) below and extract the following for EACH episode:

### Output Format

```markdown
---
## Episode: {filename}
**Numeric ID:** {numeric_id}
**Word count:** {word_count}
**Series:** {series}

### Core Message
[4-5 sentence summary of the episode's central thesis and takeaways]

### Key Training Recommendations
- [Imperative, actionable recommendation 1]
- [Imperative, actionable recommendation 2]
- ...

### Caveats & Qualifications
- [Important "it depends" factors, population-specific advice, risk warnings]
- ...

### Contrarian or Non-Obvious Claims
- [Claims that challenge conventional wisdom or common practice]
- ...

### Evolving Views
- [If the speaker references changing their mind, updating a previous recommendation, or contradicting earlier episodes — note it here with context]
- [If none apparent, write "None identified in this episode."]
```

## Extraction Rules

1. **Be specific and actionable.** "Do intervals" is too vague. "Perform 4x8min intervals at 105-110% FTP with 4min recovery" is good.
2. **Preserve numbers.** Keep specific percentages, durations, frequencies, wattages, and ranges.
3. **Attribute claims.** If the host vs. guest disagree, note who said what.
4. **Flag uncertainty.** If the speaker hedges ("probably", "we think", "the data suggests"), preserve that nuance.
5. **Skip:** Podcast logistics, ad reads, guest biography details, extended anecdotes without training takeaways, deep biochemistry pathways without practical application.
6. **Chronological context:** The numeric ID in the filename approximates chronological order. Earlier = older episodes. This matters for tracking view evolution.

## Transcripts

{transcripts}
