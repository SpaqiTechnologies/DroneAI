# Integration Tests

## Purpose
Validate real interactions:
- sensor manager discovery/polling
- fusion with multiple inputs
- planner execution against simulated environments

## How to run
```bash
pytest tests/integration
```

## Fixtures
- `mock_sensor_api()` for consistent injection