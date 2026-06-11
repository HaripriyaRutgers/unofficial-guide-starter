# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Domain

<!-- What topic or category of knowledge does your system cover?
     Why is this knowledge valuable, and why is it hard to find through official channels?
     Example: "Student reviews of CS professors at [university] — useful because official
     course descriptions don't reflect teaching style, exam difficulty, or workload." -->

The chosen domanin is career pathways related to Computer Science that are not just Software Engineering. A lot of kids fresh out of highschool, come into college undecided and eventually endup doing a Computer Science major, which they may or may not essentially enjoy. So this is for those kids out these that are "stuck" doing the major, they can find roles/career pathways do not entirely deviate from the CS degree, but are closely related such as Visual Effects in the entertainment industry or UI/UX which is making interactive user experience. This information is in my opinion hard to find through official channels because not all the universities offer a variety of courses that overlap with the CS major. The information is valuable as well since a litlle information and awareness can change thhe course of a student's life.
---

## Document Sources

<!-- List every source you collected documents from.
     Be specific: include URLs, subreddit names, forum thread titles, or file names.
     Aim for variety — sources that together cover different subtopics or perspectives. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | CS career questions subreddit|It is helpful for looking up questions about the industry from those who are entering to those that are already working in the field |https://www.reddit.com/r/cscareerquestions/ | 
| 2 | Awsome subreddits list |A big list of related subreddiys, listed by categories like programming, career, front end dev etc. | https://github.com/iCHAIT/awesome-subreddits|
| 3 | The Muse – CS Degree Jobs|Career counselor perspective on 9+ roles CS majors can pursue outside software development |https://www.themuse.com/advice/computer-science-degree-major-jobs |
| 4 |Collegewise – Alternative Pathways in CS |Discusses how skills like problem-solving and HCI travel across majors and roles |https://go.collegewise.com/alternative-pathways-to-a-career-in-computer-science |
| 5 |ScreenSkills – Careers in VFX |UK industry body with a full career map for VFX roles — covers programming-heavy roles too |https://www.screenskills.com/job-profiles/browse/visual-effects-vfx/ |
| 6 |GitHub – Awesome Cybersecurity Subreddits |Curated list of cybersecurity subreddits (r/netsec, r/cybersecurity, r/cissp, etc.) |https://github.com/d0midigi/awesome-cybersecurity-subreddits |
| 7 |CareerExplorer |Career profiles with skill and personality matches |https://www.careerexplorer.com/careers/?page=2&utm_ |
| 8 |Teamblind – "Non-SWE career opportunities for CS major" |Anonymous forum thread: CS grad with 4.0 who found SWE "boring and isolating" asks for alternatives; community replies cover PM, consulting, solutions engineering  |https://www.teamblind.com/post/non-swe-career-opportunities-for-cs-major-bh4qscys |
| 9 |Everything Technical Writing – Career Guide |Covers technical writer sub-roles (developer advocate, UX writer, API documentation, content marketer) with salaries and how to enter from a CS background | |https://www.everythingtechnicalwriting.com/everything-you-need-to-know-about-technical-writing/  |
| 10 |UChicago – Careers in Gaming |Lists every role in the games industry beyond programming: narrative designer, sound designer, game analyst, VFX artist, UX researcher, producer — shows breadth CS students don't know exists  |https://careeradvancement.uchicago.edu/careers-in/gaming/ |


---

## Chunking Strategy

<!-- Describe your chunking approach with enough specificity that someone else could reproduce it.
     Include:
     - Chunk size (characters or tokens) and why that size fits your documents
     - Overlap size and why (or why not) you used overlap
     - Any preprocessing you did before chunking (e.g., stripping HTML, removing headers)
     - What your final chunk count was across all documents -->

**Chunk size:**
400–500 tokens, counted with `tiktoken` (cl100k_base encoding) rather than approximated by word count. The 500-token figure is a hard ceiling and 400 is the target for full sliding-window chunks.

**Overlap:**
50 tokens between consecutive sliding-window chunks. I used overlap so that a sentence or idea sitting right on a chunk boundary still appears (at least partly) in both neighbors, instead of being cut in half and lost to retrieval.

**Why these choices fit your documents:**
My sources are a mix of three structures, so I did not use one strategy for everything:
- **Articles** (The Muse, Collegewise, Technical Writing, the GitHub lists) → a sliding window of 400–500 tokens with 50-token overlap. That size is big enough to hold a full role description but small enough that the embedding isn't diluted by unrelated text.
- **Forum threads** (Teamblind) → split by individual reply/post, since each reply is a self-contained opinion and the natural unit of meaning. A reply only falls back to the sliding window if it exceeds 500 tokens.
- **Career maps** (ScreenSkills) → split by role. ScreenSkills' VFX page is just a directory of role *links*, so I crawl each linked sub-page and treat one role page = one role block (windowed if it exceeds 500 tokens).
- **The Muse specifically** → split on role boundaries (each role title is followed by an "Average salary" line). The default window was gluing the tail of one job onto the next two jobs, which diluted the signal so badly that a UX query couldn't match. Splitting per role fixed it.

Preprocessing before chunking: fetched with a browser User-Agent, parsed with BeautifulSoup, stripped `<nav>/<header>/<footer>/<aside>/<script>` and any element whose class/id matched boilerplate words (matched as whole words so "ad" doesn't delete "heading"), decoded HTML entities, and for GitHub pages stripped all non-ASCII emoji/symbols and dropped 1–3 word header-only lines.

**Final chunk count:**
191 chunks across 7 successfully ingested sources (Reddit, CareerExplorer, and the UChicago gaming page did not ingest — the first two are JavaScript-rendered single-page apps that `requests` can't read, and the UChicago URL now returns 404).

---

## Embedding Model

<!-- Name the embedding model you used and explain your choice.
     Then answer: if you were deploying this system for real users and cost wasn't a constraint,
     what tradeoffs would you weigh in choosing a different model?
     Consider: context length limits, multilingual support, accuracy on domain-specific text,
     latency, and local vs. API-hosted. -->

**Model used:**
`all-MiniLM-L6-v2` via `sentence-transformers`, running locally. It produces 384-dimensional embeddings, is small (~90 MB), and embedded all 191 chunks in about 1–2 seconds on my machine with no API cost. I store the vectors in a local persistent ChromaDB collection (`cs_careers`) configured with **cosine** distance (`hnsw:space: cosine`) instead of Chroma's default squared-L2, so that distances land on an interpretable 0–2 scale where I can reason about thresholds.

**Production tradeoff reflection:**
If I were deploying this for real users and cost wasn't a constraint, I'd weigh a larger model like `all-mpnet-base-v2` or an API-hosted embedding model (e.g. OpenAI `text-embedding-3-large` or Cohere). The tradeoffs:
- **Accuracy on domain text:** MiniLM is fast but shallow — my UX-researcher query only reached a cosine distance of ~0.51 even when the right chunk ranked first. A bigger model would likely pull genuinely-relevant chunks well under 0.45 and separate them more cleanly from noise.
- **Context length:** MiniLM truncates input at 256 word-pieces, so my 400–500 token chunks are actually clipped during embedding. A model with a longer input window would embed the whole chunk, not just its front.
- **Latency vs. local control:** local MiniLM has zero network latency and no per-call cost, which is ideal for a student project. An API model adds latency and cost per query but offloads compute and is easy to scale.
- **Multilingual:** my corpus is English-only, so multilingual support isn't needed now, but an API model would handle it for free if the audience expanded.

For this project the right call was the small local model; for production I'd accept higher cost/latency for `all-mpnet-base-v2` or a hosted model to get better domain accuracy.

---

## Grounded Generation

<!-- Explain how your system enforces grounding — how does it prevent the LLM from answering
     beyond the retrieved documents?
     Describe both your system prompt (what instruction you gave the model) and any structural
     choices (e.g., how you formatted the context, whether you filtered low-relevance chunks).
     Do not just say "I told it to use the documents" — show the actual instruction or explain
     the mechanism. -->

**System prompt grounding instruction:**
I send this exact system prompt to the LLM (`llama-3.3-70b-versatile` on Groq, temperature 0):

> "You are a career advisor. Answer the user's question using ONLY the information in the provided documents below. Do not use any outside knowledge. If the documents do not contain enough information to answer the question, respond with exactly: 'I don't have enough information on that in my sources.' Always end your answer with a Sources line listing the document names you used."

Beyond the prompt text, there are structural choices that enforce grounding:
- The retrieved chunks are formatted into a **numbered context block** (`[1] source: themuse.com  text: ...`) and passed in the user message as `Documents:` directly above the question, so the model is physically given only those 5 chunks to work from.
- **Temperature is 0**, so the model is as deterministic and literal as possible rather than creatively filling gaps.
- The fixed fallback sentence gives the model a clear "escape hatch," which is why an off-topic query like "best restaurant in New York?" returns the fallback verbatim instead of hallucinating.

**How source attribution is surfaced in the response:**
Source attribution is **not** parsed from the LLM's text — it's extracted programmatically from the retrieval metadata. `ask()` takes the `source` field of each retrieved chunk and de-duplicates it while preserving rank order (`list(dict.fromkeys(...))`), then returns it as `result["sources"]`. The Gradio UI renders that list as a bullet list in a separate "Sources" box. This means the displayed sources reflect exactly which documents were retrieved and cannot be invented by the model. (Side effect: the list shows every retrieved source, which can be broader than the subset the model actually quoted.)

---

## Evaluation Report

<!-- Run your 5 test questions from planning.md through your system and record the results.
     Be honest — a partially accurate or inaccurate result that you explain well is more
     valuable than a suspiciously perfect result. -->

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 |What does a UX researcher actually do day to day? |Designs user studies, analyzes behavior data, presents findings to product teams; doesn't require coding |A UX researcher conducts research on a user's experience with a product to improve its usability. Their day-to-day work involves conducting tests to see if new features might make products easier or more enjoyable to use. They also conduct experiments and interview users through various methods to find out what they need and want. |Relevant |Accurate |
| 2 |How did someone with a CS degree break into UX design without a design background? |Sources like the Design Buddies story describe building a portfolio, joining design communities, and applying to entry-level roles after being ghosted many times |I don't have enough information on that in my sources.|Relevant|Partially Accurate|
| 3 |What VFX roles are available to someone with a CS/programming background?|Pipeline TD, rendering engineer, compositor — ScreenSkills describes technical roles alongside creative ones |1. Software developer: They produce the technology required for a visual effects (VFX) project, create systems for technical directors, and design new digital tools. Technical director (TD): Although not explicitly stated as a role for CS/programming background, software developers often work closely with TDs, and having a CS/programming background can be beneficial for this role. Compositing supervisor: While not exclusively a programming role, having a CS/programming background can be useful in this position, which involves overseeing the compositing process and working with software. Graphics programmer: This role is mentioned as a potential career path in the games industry, which can be related to VFX. | Relavant |Partially accurate|
| 4 |What is the difference between a UX designer and a UX researcher? |Designer focuses on visual/interaction output; researcher focuses on user studies, interviews, and translating findings into product decisions |The provided documents do not explicitly mention the role of a UX designer, but they do describe the role of a UX researcher. According to the document from themuse.com, a UX researcher conducts research to improve a product's usability, and their work is closely related to UX designers, but more focused on conducting experiments and interviewing users. This implies that a UX designer may be more involved in the design process, whereas a UX researcher is more focused on the research aspect. However, without more information on the role of a UX designer, it's difficult to provide a more detailed comparison.| Partially relavant|Partially accurate|
| 5 |What non-SWE careers do CS students on forums actually talk about switching to?|Product management, UX/design, solutions engineering, technical writing — based on Teamblind/Reddit thread content |Based on the provided documents, CS students on forums discuss switching to careers such as Product Management (#pm) and Consulting, as mentioned in the hashtags (#pm #consulting) at the end of the posts. |Off-target |Partially accurate|

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

<!-- Identify at least one question where retrieval or generation did not work as expected.
     Write a specific explanation of *why* it failed, tied to a part of the pipeline.

     "The answer was wrong" is not an explanation.

     "The relevant information was split across a chunk boundary, so retrieval returned
     only half the context — the model didn't have enough to answer correctly" is an explanation.

     "The embedding model treated the professor's nickname as out-of-vocabulary and returned
     results from an unrelated review" is an explanation. -->

**Question that failed:**
Question 5 — "What non-SWE careers do CS students on forums actually talk about switching to?" (marked Off-target retrieval in the table above).

**What the system returned:**
A thin answer that only said CS students discuss Product Management and Consulting, justified by the `#pm #consulting` hashtags at the end of the Teamblind post. It missed the substance of the thread (solutions engineering, technical writing, etc.) and leaned on the hashtags rather than the discussion itself.

**Root cause (tied to a specific pipeline stage):**
This is an **ingestion + chunking** failure, not a generation one. Teamblind is a JavaScript-rendered single-page app, so `requests` + BeautifulSoup only saw a partial HTML shell — most of the community replies never made it into `raw_docs/teamblind.txt`. What *did* get captured was mostly the original poster's question plus page furniture. Then, because the forum extractor + sliding window produced several **near-duplicate chunks of the same original post** (`teamblind_1`, `_2`, `_3`, `_4` are almost identical), the top-5 retrieval for this query was dominated by 4 copies of one post instead of 5 distinct perspectives. With low-diversity context, the model had little real reply content to summarize, so it fell back to the hashtags.

**What you would change to fix it:**
Two changes, both upstream of generation:
1. **Ingestion:** render Teamblind with a headless browser (Playwright/Selenium) or use an API so the actual replies are captured, instead of relying on `requests` which can't execute the page's JavaScript.
2. **Retrieval/chunking:** de-duplicate near-identical chunks before indexing (or at query time, drop results whose text is >90% similar to a higher-ranked result) so the top-5 returns 5 distinct posts rather than 4 copies of one. This would also help every forum query, not just this one.

---

## Spec Reflection

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->

**One way the spec helped you during implementation:**
Writing the chunking strategy and source list in planning.md *before* coding meant I could hand the implementation a precise target: "400–500 tokens, 50-token overlap, forum split by reply, career maps split by role." Because those numbers and rules were already decided, the chunking code came out structured around source *type* from the start (article vs. forum vs. career map) instead of one generic splitter that I'd have had to rewrite later. The explicit source table also made it obvious early which sources were forums vs. articles vs. career maps, which is exactly the distinction the chunker branches on.

**One way your implementation diverged from the spec, and why:**
The spec assumed each source was a single page I could scrape once. ScreenSkills broke that assumption: its VFX URL is just a *directory* of role links with no descriptions on the page itself, so chunking it produced 13 useless 20–40 token fragments. I diverged by making the ScreenSkills ingestion **crawl** each role's linked sub-page and treat one sub-page as one role block — which turned 13 junk fragments into 126 real chunks (avg ~444 tokens). I also had to rename `chunk.py` to `chunking.py` (a file named `chunk.py` shadows Python's stdlib `chunk` module that Gradio imports indirectly), and pin Gradio to 4.44.1 because my environment's Python 3.9 can't install the 6.9+ the spec listed. None of these were in the plan — they were forced by reality and discovered during implementation.

---

## AI Usage

<!-- Describe at least 2 specific instances where you used an AI tool during this project.
     For each: what did you give the AI as input, what did it produce, and what did you
     change, override, or direct differently?

     "I used Claude to help me code" is not sufficient.
     "I gave Claude my Chunking Strategy section from planning.md and asked it to implement
     chunk_text(). It returned a function using a fixed character split. I overrode the
     chunk size from 500 to 200 because my documents are short reviews, not long guides." -->

**Instance 1**

- *What I gave the AI:* My Chunking Strategy section from planning.md plus the list of 10 sources, and asked it to implement `ingest.py` and `chunk.py` with a from-scratch `chunk_text()` (no LangChain), using tiktoken for token counts.
- *What it produced:* A sliding-window chunker plus per-source-type extractors (forum split by reply, career map split by heading) and a validation report. The first version had two real bugs it surfaced when run: a class-filter that matched "ad"/"header" as substrings and deleted almost all article content, and a BeautifulSoup loop that crashed when it removed nodes mid-iteration.
- *What I changed or overrode:* After seeing retrieval fail, I directed two corrections it would not have made on its own: (1) ScreenSkills had to **crawl sub-pages** because the landing page held only role titles, and (2) The Muse had to be **split on role boundaries** ("Average salary" lines) because one chunk was combining UX Researcher + Product Manager + Data Scientist, which is why a UX query couldn't match. I also had it switch the class filter from substring to whole-word matching.

**Instance 2**

- *What I gave the AI:* The chunks.json format and a spec for the embedding/retrieval/generation stages — all-MiniLM-L6-v2, local persistent ChromaDB collection `cs_careers`, top-k=5, and a Groq `ask()` function with an exact grounding system prompt.
- *What it produced:* `embed.py`, `retrieve.py`, and `generate.py` with the grounded `ask()` and the Gradio UI. It pointed out that my distance scores were uninterpretable and recommended a change I accepted.
- *What I changed or overrode:* I directed it to configure ChromaDB with **cosine distance** instead of the default squared-L2 so I could reason about a 0.6 threshold, and to add a `--reset` flag to wipe the stale collection after I re-chunked. I also pushed back when it implied bad chunks alone caused my high distances — it correctly clarified that my weak UX-query scores were a **corpus coverage gap**, not a code bug, which changed how I interpreted my evaluation results.
