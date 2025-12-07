# Contributing to Judge Agent v3

This guide helps you extend and customize Judge Agent for your needs.

## Project Structure

```
src/judge_agent/
├── supervisor.py          # Main evaluation orchestrator
├── langchain.py          # Alternative LangChain implementation
├── enhanced_dimensions.py # Specialized evaluators
└── utils/                # Shared utilities
```

## Common Extensions

### 1. Connect Your Own Agent

Replace `MockAgent` with your real agent:

```python
from judge_agent.supervisor import AgentConnector

class MyLangChainAgent(AgentConnector):
    def __init__(self, chain):
        self.chain = chain
    
    async def invoke(self, prompt: str) -> str:
        # Call your agent and return the response
        result = await self.chain.ainvoke({"input": prompt})
        return result["output"]

# Use it
from my_agent import my_chain
connector = MyLangChainAgent(my_chain)
judge = JudgeAgentSupervisor(connector)
```

### 2. Add a New Evaluation Dimension

In `enhanced_dimensions.py`:

```python
class MyCustomEvaluator:
    def __init__(self, llm: ChatBedrock):
        self.llm = llm
    
    async def generate_tests(self, agent_context: str, num_tests: int) -> list[dict]:
        # Generate test cases using LLM
        ...
    
    async def evaluate(self, connector, agent_context: str) -> MyResult:
        # Run tests and return results
        ...
```

Then add to the supervisor's dimension list.

### 3. Customize Thresholds

Edit `supervisor.py`:

```python
class Config:
    SECURITY_THRESHOLD = 0.90  # Increase for stricter security
    PRIVACY_THRESHOLD = 0.90
    GENERAL_THRESHOLD = 0.75
```

Or use environment variables:

```bash
export SECURITY_PASS_THRESHOLD=0.9
```

### 4. Add Custom Prompts

Edit the `DIMENSION_PROMPTS` dict in `supervisor.py`:

```python
DIMENSION_PROMPTS[EvaluationDimension.SECURITY] = """Your custom prompt..."""
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_judge_agent.py -v

# With coverage
pytest tests/ --cov=src/judge_agent
```

## Code Style

```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/
```

## Questions?

Open an issue in the repository.
