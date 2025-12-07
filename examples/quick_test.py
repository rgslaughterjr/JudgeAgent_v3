#!/usr/bin/env python3
"""
Quick Test - Verify Judge Agent is working

This script runs a minimal evaluation using MockAgent to verify
your setup is correct. No AWS credentials required.

Usage:
    python examples/quick_test.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from judge_agent.supervisor import (
    JudgeAgentSupervisor,
    AgentConfig,
    MockAgent,
    EvaluationDimension
)


async def main():
    print("🏆 Judge Agent v3 - Quick Test")
    print("=" * 50)
    
    # Configure a test agent
    config = AgentConfig(
        agent_id="quick-test-001",
        name="Test Agent",
        framework="mock",
        risk_level="low",  # Low = fewer tests for speed
        description="Quick test to verify setup",
        data_access=[]
    )
    
    print(f"Testing: {config.name}")
    print(f"Risk Level: {config.risk_level}")
    print()
    
    # Use mock agent (no real LLM calls)
    connector = MockAgent()
    judge = JudgeAgentSupervisor(connector)
    
    print("Running evaluation (this may take 1-2 minutes)...")
    print()
    
    try:
        result = await judge.evaluate_parallel(config)
        
        print("=" * 50)
        print("RESULTS")
        print("=" * 50)
        print(f"Overall Score: {result['overall_score']:.1%}")
        print(f"Production Ready: {'✅ YES' if result['passes_gate'] else '❌ NO'}")
        print()
        print("Dimension Scores:")
        for dim in result['dimension_results']:
            status = "✅" if dim['passed'] else "❌"
            print(f"  {status} {dim['dimension']}: {dim['score']:.1%}")
        
        print()
        print("✅ Setup verified! Judge Agent is working correctly.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure you're in the JudgeAgent_v3 directory")
        print("2. Run: pip install -r requirements.txt")
        print("3. Check AWS credentials: aws configure")
        raise


if __name__ == "__main__":
    asyncio.run(main())
