# Solve the zodiac alignment episode

Seven symbolic bodies occupy positions `0..11` on a cyclic ring. Move every
body to its target using the fewest actions possible.

Inspect the scenario:

```bash
astro-task observe
```

An action is `2 * body_index + direction`, where an even action moves that body
counterclockwise by one and an odd action moves it clockwise by one. Positions
wrap modulo 12. You may evaluate a candidate trajectory without changing state:

```bash
astro-task evaluate '[1, 1, 2]'
```

Write the final trajectory to `/app/answer.json`:

```json
{"actions": [1, 1, 2]}
```

Only the `actions` key is allowed. At most 64 actions may be submitted. The
verifier gives partial credit for reducing total circular distance and full
credit only for an optimal successful trajectory.
