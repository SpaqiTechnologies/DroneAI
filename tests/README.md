# Tests

## Structure
- `unit`: component correctness in isolation
- `integration`: real interactions with simulated environments
- `system`: end-to-end stack validation

## Run all
```bash
tox -e test
```

## Coverage
```bash
coverage run -m pytest && coverage html