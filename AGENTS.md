# Agent notes

Research feature families (F2, F3A, F3B, F4, price, inventory, commercial) go through `pkg.research.harness`. Read `.cursor/skills/research-feature-family/SKILL.md` before adding a family.

Never mutate the frozen benchmark, `XGB_PARAMS`, or F0 feature lists. Write only to a new `src/data/results/{family}/` and `docs/{family}_*.md`. Do not overwrite F0/F1/F2/F3A artifacts or start the next family in the same change.
