# Solve the transit timing episode

Three symbolic bodies move around positions `0..11` on a cyclic ring. Schedule
when to resolve each task so its body is as close as possible to its target.
Only one action may be taken per day.

Inspect the scenario:

```bash
astro-task observe
```

Action `0` waits. Action `i + 1` resolves task `i`. A resolved task earns
`1 - circular_distance / 6`; waiting costs `0.02`. After every action, each
position advances by its corresponding transit speed modulo 12. You may
evaluate a complete candidate trajectory without changing state:

```bash
astro-task evaluate '[0, 0, 1, 2, 3]'
```

Write the final trajectory to `/app/answer.json`:

```json
{"actions": [0, 0, 1, 2, 3]}
```

Only the `actions` key is allowed. At most 24 actions may be submitted. The
verifier normalizes the achieved return against the optimal return and gives
full credit only for the optimal schedule.
