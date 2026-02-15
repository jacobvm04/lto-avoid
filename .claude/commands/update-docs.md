Review recent code changes and ensure all user-facing documentation is accurate and up to date.

## Steps

1. **Identify what changed.** Run `git diff` and `git log` to understand recent changes — new modules, renamed functions, changed APIs, new features, removed functionality, changed behavior.

2. **Audit existing docs.** Read all user-facing documentation files:
   - `README.md` — project overview, install instructions, usage examples, API summary
   - Any files in `examples/` or `docs/`
   - `CLAUDE.md` — developer workflow docs (update if dev practices changed)
   - Docstrings on public API in `src/lto_avoid/__init__.py`

3. **Check each doc against the code.** For every claim in the docs, verify it's still true:
   - Do code examples still work? Do imports match `__all__`?
   - Are function signatures and parameter names accurate?
   - Are install instructions correct (`uv sync`, dependencies)?
   - Are architecture descriptions still accurate?
   - Are any new public functions/features undocumented?

4. **Fix stale docs.** Update anything that's wrong or outdated. Be precise — match actual function names, parameter names, and behavior.

5. **Add new entries if warranted.** If significant new functionality was added that users would want to know about, add documentation for it. Don't over-document internal details — focus on what a user of the library needs.

6. **Create missing docs if needed.** If there's no README.md yet and the project is at a point where one would be useful, create one. If there are non-trivial usage patterns that deserve an example, add one.

7. **Don't invent.** Only document what actually exists in the code. Don't add aspirational features or TODO sections.
