# CLAUDE.md — Image Embedding Lab

Read **[AGENTS.md](AGENTS.md)** before doing anything — it is the source of truth for this
repo's measurement, caching, design, and workflow constraints, each traced to the commit where
violating it caused a real bug.

The non-negotiables, in one breath: pool embeddings before comparing; feed each encoder its
native resolution (Chrome canvas resampling is not innocent); no similarity number without its
different-image floor; browser and CLI share `lib/` and the same transformers.js version;
never delete or duplicate `transformers-cache`; verify the same numbers in Node and the browser
before pushing; errors on the page, not the console; match the existing single-file page style.
