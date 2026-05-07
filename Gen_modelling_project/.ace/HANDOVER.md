# Ace handover format (v1)

When the user requests a handover for AceBrain, generate exactly this block, verbatim, with the placeholder values replaced. No commentary outside the block.

=== ACE-HANDOVER v1 ===
domain: {ai-research}
project-slug: {slug}
project-path: {absolute-path}
session-date: {YYYY-MM-DD}
session-summary: |
  - {3-5 bullets}
new-knowledge:
  - "{concept: one-line summary}"
papers-referenced:
  - title: "{title}"
    arxiv: {id}
    relevance: {one-line}
tools-or-libraries-used:
  - {name}
open-questions:
  - "{question}" [{priority-tag}]
artefacts-to-ingest (newly-created-files-or-folders-in-this-session):
  - {relative-path-from-repo-root}
next-suggested-step: |
  {one-paragraph}
=== END ACE-HANDOVER ===

Rules:
- Only items genuinely new in this session.
- `domain` must be one of: ai-research, deep-learning, ai-safety, ml-systems, finance-investing.
- `priority-tag` ∈ {urgent, exam-relevant, application-relevant, curiosity}.
