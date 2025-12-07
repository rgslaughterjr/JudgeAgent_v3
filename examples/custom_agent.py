#!/usr/bin/env python3
"""
Custom Agent Example - Connect your own agent to Judge Agent

This example shows how to create a connector for your own agent
and run a full evaluation.

Usage:
    python examples/custom_agent.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from judge_agent.supervisor import (
    JudgeAgentSupervisor,
    AgentConnector,
    AgentConfig,
)


# =============================================================================
# STEP 1: Implement your agent connector
# =============================================================================

class MyCustomAgent(AgentConnector):
    """
    Example connector for a custom agent.
    
    Replace the invoke() method with your actual agent logic.
    """
    
    def __init__(self, model_name: str = "my-model"):
        self.model_name = model_name
        # Initialize your agent here
        # self.client = MyAgentClient(...)
    
    async def invoke(self, prompt: str) -> str:
        """
        Process a prompt and return the agent's response.
        
        This is where you connect to your actual agent.
        """
        # Example: Call your LangChain chain
        # response = await self.chain.ainvoke({"input": prompt})
        # return response["output"]
        
        # Example: Call an API
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(self.api_url, json={"prompt": prompt}) as resp:
        #         data = await resp.json()
        #         return data["response"]
        
        # Placeholder - replace with your logic
        return f"Response from {self.model_name}: I processed your request about '{prompt[:50]}...'"


# =============================================================================
# STEP 2: Configure and run evaluation
# =============================================================================

async def main():
    print("🏆 Judge Agent v3 - Custom Agent Example")
    print("=" * 50)
    
    # Configure the agent being evaluated
    config = AgentConfig(
        agent_id="my-custom-agent-001",
        name="My Custom Agent",
        framework="custom",
        risk_level="medium",
        description="Custom agent for demonstration",
        data_access=["database", "api"]
    )
    
    # Create your connector
    connector = MyCustomAgent(model_name="gpt-4")
    
    # Create the judge
    judge = JudgeAgentSupervisor(connector)
    
    print(f"Evaluating: {config.name}")
    print(f"Risk Level: {config.risk_level}")
    print()
    print("Running evaluation...")
    
    # Run evaluation
    result = await judge.evaluate_parallel(
        config, 
        evaluator_user="developer@company.com"
    )
    
    # Display results
    print()
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Overall Score: {result['overall_score']:.1%}")
    print(f"Production Ready: {'✅ YES' if result['passes_gate'] else '❌ NO'}")
    print()
    
    print("Dimension Breakdown:")
    for dim in result['dimension_results']:
        status = "✅" if dim['passed'] else "❌"
        print(f"  {status} {dim['dimension']}: {dim['score']:.1%} ({dim['tests_run']} tests)")
    
    # Show any errors
    if result.get('error_tracking'):
        print()
        print("Errors encountered:")
        for dim, errors in result['error_tracking'].items():
            print(f"  {dim}: {len(errors)} errors")


if __name__ == "__main__":
    asyncio.run(main())
