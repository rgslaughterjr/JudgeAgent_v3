"""Test audit logging functionality"""

import asyncio
import json
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from uuid import uuid4

from utils.audit_log import AuditLogger, AuditLogEntry, create_audit_logger


async def test_audit_log_entry():
    """Test AuditLogEntry serialization"""
    print("=" * 60)
    print("Testing AuditLogEntry")
    print("=" * 60)

    entry = AuditLogEntry(
        timestamp="2025-12-06T12:00:00Z",
        evaluation_id="eval-123",
        agent_id="agent-001",
        agent_name="Test Agent",
        evaluator_user="test@example.com",
        framework="langchain",
        risk_level="medium",
        overall_score=0.85,
        passed=True,
        dimension_scores={
            "security": {"score": 0.9, "passed": True, "test_count": 10},
            "privacy": {"score": 0.8, "passed": True, "test_count": 8}
        },
        critical_failures=0,
        deployment_stage="development",
        metadata={"version": "1.0"}
    )

    # Test serialization
    json_str = entry.to_json()
    assert isinstance(json_str, str), "Should serialize to string"
    assert "eval-123" in json_str, "Should contain evaluation ID"
    print(f"[PASS] Serialization: {len(json_str)} chars")

    # Test deserialization
    parsed = AuditLogEntry.from_json(json_str)
    assert parsed.evaluation_id == entry.evaluation_id, "Should deserialize correctly"
    assert parsed.overall_score == entry.overall_score, "Should preserve score"
    assert parsed.dimension_scores == entry.dimension_scores, "Should preserve dimension scores"
    print(f"[PASS] Deserialization")

    print()


async def test_local_logging():
    """Test local file logging"""
    print("=" * 60)
    print("Testing Local File Logging")
    print("=" * 60)

    # Create temporary log directory
    test_log_dir = Path("./test-audit-logs")
    if test_log_dir.exists():
        shutil.rmtree(test_log_dir)

    logger = AuditLogger(
        local_log_dir=str(test_log_dir),
        enable_s3=False
    )

    # Log multiple evaluations
    evaluation_ids = []
    for i in range(3):
        eval_id = str(uuid4())
        evaluation_ids.append(eval_id)

        await logger.log_evaluation(
            evaluation_id=eval_id,
            agent_id=f"agent-{i % 2}",  # Two different agents
            agent_name=f"Test Agent {i}",
            evaluator_user="test@example.com",
            framework="langchain",
            risk_level="medium",
            overall_score=0.7 + (i * 0.1),
            passed=i >= 1,  # First one fails
            dimension_scores={
                "security": {"score": 0.8, "passed": True, "test_count": 10}
            },
            critical_failures=1 if i == 0 else 0,
            deployment_stage="development"
        )

    print(f"[PASS] Logged {len(evaluation_ids)} evaluations")

    # Verify files were created
    today = datetime.now(timezone.utc)
    log_path = logger._get_local_log_path(today)
    assert log_path.exists(), f"Log file should exist: {log_path}"
    print(f"[PASS] Log file created: {log_path}")

    # Verify JSONL format
    lines = log_path.read_text().splitlines()
    assert len(lines) == 3, f"Should have 3 lines, got {len(lines)}"

    for line in lines:
        data = json.loads(line)
        assert 'evaluation_id' in data, "Should have evaluation_id"
        assert 'overall_score' in data, "Should have overall_score"

    print(f"[PASS] JSONL format verified")

    # Cleanup
    shutil.rmtree(test_log_dir)
    print()


async def test_query_history():
    """Test querying audit history"""
    print("=" * 60)
    print("Testing Query History")
    print("=" * 60)

    # Create temporary log directory
    test_log_dir = Path("./test-audit-logs")
    if test_log_dir.exists():
        shutil.rmtree(test_log_dir)

    logger = AuditLogger(
        local_log_dir=str(test_log_dir),
        enable_s3=False
    )

    # Log evaluations for different agents
    agent_ids = ["agent-alpha", "agent-beta"]
    now = datetime.now(timezone.utc)

    for i in range(5):
        await logger.log_evaluation(
            evaluation_id=str(uuid4()),
            agent_id=agent_ids[i % 2],
            agent_name=f"Agent {agent_ids[i % 2]}",
            evaluator_user="test@example.com",
            framework="langchain",
            risk_level="medium",
            overall_score=0.6 + (i * 0.05),
            passed=i >= 2,
            dimension_scores={},
            critical_failures=0,
            deployment_stage="development"
        )

    # Query all entries
    all_entries = await logger.query_history()
    assert len(all_entries) == 5, f"Should find 5 entries, found {len(all_entries)}"
    print(f"[PASS] Query all: {len(all_entries)} entries")

    # Query by agent_id
    alpha_entries = await logger.query_history(agent_id="agent-alpha")
    assert len(alpha_entries) == 3, f"Should find 3 alpha entries, found {len(alpha_entries)}"
    assert all(e.agent_id == "agent-alpha" for e in alpha_entries), "Should only return alpha"
    print(f"[PASS] Query by agent_id: {len(alpha_entries)} entries")

    # Query passed only
    passed_entries = await logger.query_history(passed_only=True)
    assert len(passed_entries) == 3, f"Should find 3 passed entries, found {len(passed_entries)}"
    assert all(e.passed for e in passed_entries), "Should only return passed"
    print(f"[PASS] Query passed only: {len(passed_entries)} entries")

    # Query by min_score
    high_score_entries = await logger.query_history(min_score=0.75)
    assert all(e.overall_score >= 0.75 for e in high_score_entries), "Should filter by score"
    print(f"[PASS] Query by min_score: {len(high_score_entries)} entries")

    # Query by date range
    start_date = now - timedelta(hours=1)
    end_date = now + timedelta(hours=1)
    date_entries = await logger.query_history(
        start_date=start_date,
        end_date=end_date
    )
    assert len(date_entries) == 5, "Should find all entries in date range"
    print(f"[PASS] Query by date range: {len(date_entries)} entries")

    # Cleanup
    shutil.rmtree(test_log_dir)
    print()


async def test_agent_statistics():
    """Test agent statistics calculation"""
    print("=" * 60)
    print("Testing Agent Statistics")
    print("=" * 60)

    # Create temporary log directory
    test_log_dir = Path("./test-audit-logs")
    if test_log_dir.exists():
        shutil.rmtree(test_log_dir)

    logger = AuditLogger(
        local_log_dir=str(test_log_dir),
        enable_s3=False
    )

    # Log evaluations with varying scores
    agent_id = "agent-stats-test"
    scores = [0.5, 0.7, 0.85, 0.9, 0.95]

    for i, score in enumerate(scores):
        await logger.log_evaluation(
            evaluation_id=str(uuid4()),
            agent_id=agent_id,
            agent_name="Test Agent",
            evaluator_user="test@example.com",
            framework="langchain",
            risk_level="high",
            overall_score=score,
            passed=score >= 0.7,
            dimension_scores={},
            critical_failures=0,
            deployment_stage="staging"
        )
        # Small delay to ensure different timestamps
        await asyncio.sleep(0.01)

    # Get statistics
    stats = await logger.get_agent_statistics(agent_id, days=30)

    print(f"Statistics for {agent_id}:")
    print(f"  Total evaluations: {stats['total_evaluations']}")
    print(f"  Passed evaluations: {stats['passed_evaluations']}")
    print(f"  Pass rate: {stats['pass_rate']:.2%}")
    print(f"  Average score: {stats['average_score']:.2f}")
    print(f"  Score range: {stats['min_score']:.2f} - {stats['max_score']:.2f}")
    print(f"  Latest score: {stats['latest_score']:.2f}")

    assert stats['total_evaluations'] == 5, "Should count all evaluations"
    assert stats['passed_evaluations'] == 4, "Should count passed (score >= 0.7)"
    assert stats['pass_rate'] == 0.8, "Pass rate should be 80%"
    assert abs(stats['average_score'] - 0.78) < 0.01, "Average should be ~0.78"
    assert stats['min_score'] == 0.5, "Min score should be 0.5"
    assert stats['max_score'] == 0.95, "Max score should be 0.95"
    assert stats['latest_score'] == 0.95, "Latest should be last logged (0.95)"

    print(f"[PASS] All statistics correct")

    # Cleanup
    shutil.rmtree(test_log_dir)
    print()


async def test_date_partitioning():
    """Test date-based partitioning"""
    print("=" * 60)
    print("Testing Date Partitioning")
    print("=" * 60)

    # Create temporary log directory
    test_log_dir = Path("./test-audit-logs")
    if test_log_dir.exists():
        shutil.rmtree(test_log_dir)

    logger = AuditLogger(
        local_log_dir=str(test_log_dir),
        enable_s3=False
    )

    # Test partition path generation
    dt = datetime(2025, 12, 6, 15, 30, 0, tzinfo=timezone.utc)
    partition = logger._get_date_partition(dt)
    assert partition == "2025/12/06", f"Partition should be 2025/12/06, got {partition}"
    print(f"[PASS] Date partition format: {partition}")

    # Test local log path
    log_path = logger._get_local_log_path(dt)
    expected = test_log_dir / "2025" / "12" / "06" / "evaluations.jsonl"
    assert log_path == expected, f"Log path should be {expected}, got {log_path}"
    print(f"[PASS] Local log path: {log_path}")

    # Test S3 key
    s3_key = logger._get_s3_key(dt)
    assert s3_key == "audit-logs/2025/12/06/evaluations.jsonl", f"S3 key incorrect: {s3_key}"
    print(f"[PASS] S3 key: {s3_key}")

    # Cleanup
    shutil.rmtree(test_log_dir, ignore_errors=True)
    print()


async def test_convenience_function():
    """Test create_audit_logger convenience function"""
    print("=" * 60)
    print("Testing Convenience Function")
    print("=" * 60)

    # Test local-only mode
    logger = create_audit_logger(local_only=True)
    assert logger.enable_local, "Should enable local storage"
    assert not logger.enable_s3, "Should disable S3 in local-only mode"
    print(f"[PASS] Local-only mode")

    # Test with S3 bucket
    logger = create_audit_logger(s3_bucket="test-bucket")
    assert logger.s3_bucket == "test-bucket", "Should set S3 bucket"
    print(f"[PASS] S3 bucket configuration")

    print()


async def run_all_tests():
    """Run all audit log tests"""
    print("\n" + "=" * 60)
    print("Audit Logger Test Suite")
    print("=" * 60 + "\n")

    await test_audit_log_entry()
    await test_local_logging()
    await test_query_history()
    await test_agent_statistics()
    await test_date_partitioning()
    await test_convenience_function()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
