# Zodiac Alignment

Harbor task `Nandinitalwar/zodiac-alignment`.

The agent inspects a fixed seven-body scenario, submits an action trajectory in
`/app/answer.json`, and receives a deterministic reward in `[0, 1]`. Partial
distance reduction earns up to `0.5`; a successful trajectory earns
`0.5 + 0.5 * efficiency`, where an optimal 25-step solution scores `1.0`.

```bash
harbor run -p tasks/zodiac-alignment -a oracle
harbor run -p tasks/zodiac-alignment -a nop
```
