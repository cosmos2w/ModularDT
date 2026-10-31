# Model explanation diagram sources

These four text sources accompany the implementation-bound [Model_Explain.md](../../../Model_Explain.md). The article embeds each source as a native Mermaid block so that GitHub can render the complete explanation without numerical exports or local-only images.

| Source | Article figure | Purpose |
| --- | --- | --- |
| [01_information_flow.mmd](01_information_flow.mmd) | 1 | M/E/C inputs, shared preparation, directory and member-value paths, query field. |
| [02_exchange_block.mmd](02_exchange_block.mmd) | 2 | Ordered MM, EM, EE and ME stages with E-state carry and role feed-forward updates. |
| [03_score_to_directory.mmd](03_score_to_directory.mmd) | 3 | Raw scores, softmax preference, applied gates and prepared E-address directory. |
| [04_query_reading.mmd](04_query_reading.mmd) | 4 | Coverage, selected-member union, reusable scores/messages, coalescence and joint decoding. |

Keep the `.mmd` sources and article blocks identical, in figure order. Update the English article and the ignored local Chinese mirror together, preserving equations, symbols, code examples, figure sources and implementation binding. The notation table in the article is the authority for future model explanations; source-code names are mapped there rather than silently redefining symbols.

Use compact top-to-bottom layouts with side branches where they clarify carried states or shared quantities. Keep node labels short and put explanations in captions. Diagram edges represent information flow, not physical causality. The P95 and overlapping-address arithmetic examples in the article are explicitly synthetic.

Exported previews and PDF masters for local review belong in the ignored DATA2 documentation workspace and its common R15 report subdirectory. This directory contains maintained textual figure sources only; numerical arrays, browser caches, one-time renderers and generated images are not uploaded.
