# Pass 2: Synthesis Prompt

You are synthesizing extracted training advice from the Empirical Cycling Podcast into a thematic summary for the topic: **"{theme_name}"**.

## Instructions

Read the extraction file below. It contains per-episode extractions from multiple podcast episodes, all related to this theme. Your job is to synthesize these into a coherent thematic summary with multiple declarative sub-sections.

## Output Format

```markdown
# {theme_name}

## [Declarative, Specific Headline About a Key Finding]

[200-300 words: Start with the recommendation. Then provide supporting detail, context, and reasoning. End with caveats and "it depends" factors.]

**Recommendations:**
- [Specific, actionable bullet point]
- [Another specific recommendation]
- ...

**Sources:** Episodes {list of source episode filenames}

---

## [Another Declarative, Specific Headline]

[200-300 words]

**Recommendations:**
- ...

**Sources:** Episodes {list}

---

[...repeat for 5-10 sub-sections depending on material volume...]

---

## Evolution of Views

[How recommendations on this topic shifted over the life of the podcast. Reference specific episodes by name and numeric ID to show chronological progression. Note where the hosts changed their minds, refined recommendations, or contradicted earlier advice.]
```

## Synthesis Rules

1. **Headlines must be declarative and specific.** NOT "About Threshold Training" — instead "Sweet Spot Volume Has Diminishing Returns Beyond Two Sessions Per Week". Each headline should make a claim.
2. **200-300 words per section.** Enough depth to be useful, not so much it becomes rambling.
3. **Every recommendation must cite source episode(s).** Use episode filenames so they can be traced back.
4. **Reconcile contradictions.** If different episodes say different things, explain the nuance — don't just pick one. Note if views evolved over time.
5. **Preserve caveats.** "It depends" is often the real answer. Include the conditions under which advice changes.
6. **Merge duplicates.** If multiple episodes all say the same thing, synthesize into one section citing all of them, don't repeat.
7. **Group logically.** Related recommendations should be in the same sub-section.
8. **Aim for 5-10 sub-sections** depending on how much material exists. Fewer if the cluster is small; more if it's large and varied.

## Extraction File

{extraction_content}
