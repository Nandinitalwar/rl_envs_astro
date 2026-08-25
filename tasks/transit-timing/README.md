# Transit Timing

Harbor task `Nandinitalwar/transit-timing`.

The agent inspects a deterministic transit schedule, submits an action
trajectory in `/app/answer.json`, and receives a normalized reward in `[0, 1]`.
The optimal raw return is `2.66`; incomplete schedules are capped at `0.5`.

```bash
harbor run -p tasks/transit-timing -a oracle
harbor run -p tasks/transit-timing -a nop
```
