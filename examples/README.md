# Examples

This directory contains runnable example scripts to help you get started.

## Available Examples

### `quick_test.py`

Verify your setup is working correctly. Uses MockAgent, no AWS credentials required.

```bash
python examples/quick_test.py
```

### `custom_agent.py`

Shows how to connect your own agent to Judge Agent for evaluation.

```bash
python examples/custom_agent.py
```

## Running Examples

All examples should be run from the repository root:

```bash
cd JudgeAgent_v3
python examples/quick_test.py
```

The examples automatically add `src/` to the Python path.
