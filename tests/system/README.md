# System Tests

## Purpose
End-to-end validation of the complete drone stack:
- perception (sensors)
- planning
- control
- actuation simulation

## How to run
```bash
pytest tests/system
```

## Artifacts
- flight logs for regression comparison
- telemetry captures