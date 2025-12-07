"""Audit logging for Judge Agent evaluations"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import uuid4
from dataclasses import dataclass, asdict

try:
    import boto3
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    logging.warning("boto3 not available - S3 storage disabled")

logger = logging.getLogger(__name__)


@dataclass
class AuditLogEntry:
    """Single audit log entry"""
    timestamp: str  # ISO8601 format
    evaluation_id: str  # UUID
    agent_id: str
    agent_name: str
    evaluator_user: str  # User who triggered evaluation
    framework: str
    risk_level: str
    overall_score: float
    passed: bool
    dimension_scores: Dict[str, Dict[str, Any]]  # {dimension: {score, passed, test_count}}
    critical_failures: int
    deployment_stage: str
    metadata: Dict[str, Any] = None  # Additional metadata

    def to_json(self) -> str:
        """Convert to JSON string for JSONL format"""
        data = asdict(self)
        return json.dumps(data, separators=(',', ':'))  # Compact JSON

    @classmethod
    def from_json(cls, json_str: str) -> 'AuditLogEntry':
        """Parse from JSON string"""
        data = json.loads(json_str)
        return cls(**data)


class AuditLogger:
    """
    Audit logger for Judge Agent evaluations.

    Stores logs in JSONL format (one JSON object per line) for:
    - Immutability and append-only semantics
    - Easy line-by-line parsing
    - Compatibility with log processing tools

    Supports both local file storage and S3 with date partitioning.

    Examples:
        >>> logger = AuditLogger(s3_bucket='my-audit-logs', s3_prefix='judge-agent')
        >>> await logger.log_evaluation(
        ...     agent_id='agent-123',
        ...     agent_name='Customer Service Bot',
        ...     overall_score=0.85,
        ...     passed=True,
        ...     dimension_scores={...}
        ... )

        >>> # Query history
        >>> entries = await logger.query_history(
        ...     agent_id='agent-123',
        ...     start_date=datetime(2025, 1, 1),
        ...     end_date=datetime(2025, 12, 31)
        ... )
    """

    def __init__(
        self,
        local_log_dir: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "audit-logs",
        aws_region: str = "us-east-1",
        enable_local: bool = True,
        enable_s3: bool = True
    ):
        """
        Initialize audit logger.

        Args:
            local_log_dir: Local directory for audit logs (default: ./audit-logs)
            s3_bucket: S3 bucket name for audit logs
            s3_prefix: S3 key prefix (default: audit-logs)
            aws_region: AWS region for S3
            enable_local: Enable local file logging
            enable_s3: Enable S3 logging
        """
        self.local_log_dir = Path(local_log_dir or "./audit-logs")
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.aws_region = aws_region
        self.enable_local = enable_local
        self.enable_s3 = enable_s3 and S3_AVAILABLE and s3_bucket

        # Create local directory if needed
        if self.enable_local:
            self.local_log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize S3 client
        self.s3_client = None
        if self.enable_s3:
            self.s3_client = boto3.client('s3', region_name=self.aws_region)

    def _get_date_partition(self, dt: datetime) -> str:
        """
        Get date partition path in format: YYYY/MM/DD

        Args:
            dt: Datetime to partition

        Returns:
            Partition path string
        """
        return f"{dt.year:04d}/{dt.month:02d}/{dt.day:02d}"

    def _get_local_log_path(self, dt: datetime) -> Path:
        """
        Get local log file path with date partitioning.

        Args:
            dt: Datetime for partition

        Returns:
            Path to log file
        """
        partition = self._get_date_partition(dt)
        log_dir = self.local_log_dir / partition
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / "evaluations.jsonl"

    def _get_s3_key(self, dt: datetime) -> str:
        """
        Get S3 key with date partitioning.

        Args:
            dt: Datetime for partition

        Returns:
            S3 key string
        """
        partition = self._get_date_partition(dt)
        return f"{self.s3_prefix}/{partition}/evaluations.jsonl"

    async def log_evaluation(
        self,
        evaluation_id: str,
        agent_id: str,
        agent_name: str,
        evaluator_user: str,
        framework: str,
        risk_level: str,
        overall_score: float,
        passed: bool,
        dimension_scores: Dict[str, Dict[str, Any]],
        critical_failures: int = 0,
        deployment_stage: str = "development",
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditLogEntry:
        """
        Log an evaluation to audit trail.

        Args:
            evaluation_id: Unique evaluation ID (UUID)
            agent_id: Agent being evaluated
            agent_name: Human-readable agent name
            evaluator_user: User who triggered evaluation
            framework: Agent framework (langchain, bedrock, etc.)
            risk_level: Risk level (low, medium, high, critical)
            overall_score: Overall evaluation score (0-1)
            passed: Whether evaluation passed
            dimension_scores: Scores by dimension
            critical_failures: Number of critical failures
            deployment_stage: Deployment stage (development, staging, production)
            metadata: Additional metadata

        Returns:
            Created audit log entry
        """
        timestamp = datetime.now(timezone.utc)

        entry = AuditLogEntry(
            timestamp=timestamp.isoformat(),
            evaluation_id=evaluation_id,
            agent_id=agent_id,
            agent_name=agent_name,
            evaluator_user=evaluator_user,
            framework=framework,
            risk_level=risk_level,
            overall_score=overall_score,
            passed=passed,
            dimension_scores=dimension_scores,
            critical_failures=critical_failures,
            deployment_stage=deployment_stage,
            metadata=metadata or {}
        )

        json_line = entry.to_json() + "\n"

        # Write to local file
        if self.enable_local:
            await self._write_local(timestamp, json_line)

        # Write to S3
        if self.enable_s3:
            await self._write_s3(timestamp, json_line)

        logger.info(
            f"Audit log written: evaluation_id={evaluation_id}, "
            f"agent_id={agent_id}, score={overall_score:.2f}, passed={passed}"
        )

        return entry

    async def _write_local(self, timestamp: datetime, json_line: str):
        """Write to local JSONL file"""
        try:
            log_path = self._get_local_log_path(timestamp)

            # Append to file (JSONL format)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: log_path.open('a').write(json_line)
            )

            logger.debug(f"Written to local log: {log_path}")

        except Exception as e:
            logger.error(f"Failed to write local audit log: {e}")
            raise

    async def _write_s3(self, timestamp: datetime, json_line: str):
        """Write to S3 (append to existing file or create new)"""
        try:
            s3_key = self._get_s3_key(timestamp)

            # Download existing file if it exists
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
                )
                existing_content = response['Body'].read().decode('utf-8')
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    existing_content = ""
                else:
                    raise

            # Append new line
            new_content = existing_content + json_line

            # Upload back to S3
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=s3_key,
                    Body=new_content.encode('utf-8'),
                    ContentType='application/x-ndjson',
                    ServerSideEncryption='AES256'
                )
            )

            logger.debug(f"Written to S3: s3://{self.s3_bucket}/{s3_key}")

        except Exception as e:
            logger.error(f"Failed to write S3 audit log: {e}")
            raise

    async def query_history(
        self,
        agent_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        passed_only: bool = False,
        min_score: Optional[float] = None
    ) -> List[AuditLogEntry]:
        """
        Query audit log history with filters.

        Args:
            agent_id: Filter by agent ID
            start_date: Filter by start date (inclusive)
            end_date: Filter by end date (inclusive)
            passed_only: Only return passed evaluations
            min_score: Minimum score filter

        Returns:
            List of matching audit log entries
        """
        entries = []

        # Determine date range to scan
        if start_date and end_date:
            date_range = self._get_date_range(start_date, end_date)
        else:
            # Scan all available logs
            date_range = self._scan_available_partitions()

        # Read from local or S3
        if self.enable_local:
            entries.extend(await self._query_local(date_range))
        elif self.enable_s3:
            entries.extend(await self._query_s3(date_range))

        # Apply filters
        filtered = entries

        if agent_id:
            filtered = [e for e in filtered if e.agent_id == agent_id]

        if start_date:
            filtered = [
                e for e in filtered
                if datetime.fromisoformat(e.timestamp) >= start_date.replace(tzinfo=timezone.utc)
            ]

        if end_date:
            filtered = [
                e for e in filtered
                if datetime.fromisoformat(e.timestamp) <= end_date.replace(tzinfo=timezone.utc)
            ]

        if passed_only:
            filtered = [e for e in filtered if e.passed]

        if min_score is not None:
            filtered = [e for e in filtered if e.overall_score >= min_score]

        # Sort by timestamp descending (newest first)
        filtered.sort(key=lambda e: e.timestamp, reverse=True)

        return filtered

    def _get_date_range(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Generate list of dates between start and end"""
        from datetime import timedelta

        dates = []
        current = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = end_date.replace(hour=0, minute=0, second=0, microsecond=0)

        while current <= end:
            dates.append(current)
            current = current + timedelta(days=1)

        return dates

    def _scan_available_partitions(self) -> List[datetime]:
        """Scan for available log partitions"""
        # For simplicity, return last 30 days
        from datetime import timedelta
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)
        return self._get_date_range(start, end)

    async def _query_local(self, date_range: List[datetime]) -> List[AuditLogEntry]:
        """Query local log files"""
        entries = []

        for dt in date_range:
            log_path = self._get_local_log_path(dt)

            if not log_path.exists():
                continue

            try:
                loop = asyncio.get_event_loop()
                lines = await loop.run_in_executor(
                    None,
                    lambda: log_path.read_text().splitlines()
                )

                for line in lines:
                    if line.strip():
                        try:
                            entry = AuditLogEntry.from_json(line)
                            entries.append(entry)
                        except Exception as e:
                            logger.warning(f"Failed to parse audit log line: {e}")

            except Exception as e:
                logger.warning(f"Failed to read log file {log_path}: {e}")

        return entries

    async def _query_s3(self, date_range: List[datetime]) -> List[AuditLogEntry]:
        """Query S3 log files"""
        entries = []

        for dt in date_range:
            s3_key = self._get_s3_key(dt)

            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
                )

                content = response['Body'].read().decode('utf-8')
                lines = content.splitlines()

                for line in lines:
                    if line.strip():
                        try:
                            entry = AuditLogEntry.from_json(line)
                            entries.append(entry)
                        except Exception as e:
                            logger.warning(f"Failed to parse audit log line: {e}")

            except ClientError as e:
                if e.response['Error']['Code'] != 'NoSuchKey':
                    logger.warning(f"Failed to read S3 log {s3_key}: {e}")

        return entries

    async def get_agent_statistics(self, agent_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get statistical summary for an agent.

        Args:
            agent_id: Agent to analyze
            days: Number of days to analyze

        Returns:
            Statistics dictionary
        """
        from datetime import timedelta

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        entries = await self.query_history(
            agent_id=agent_id,
            start_date=start_date,
            end_date=end_date
        )

        if not entries:
            return {
                'agent_id': agent_id,
                'period_days': days,
                'total_evaluations': 0,
                'message': 'No evaluations found'
            }

        scores = [e.overall_score for e in entries]
        passed_count = sum(1 for e in entries if e.passed)

        return {
            'agent_id': agent_id,
            'period_days': days,
            'total_evaluations': len(entries),
            'passed_evaluations': passed_count,
            'pass_rate': passed_count / len(entries) if entries else 0,
            'average_score': sum(scores) / len(scores) if scores else 0,
            'min_score': min(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'latest_evaluation': entries[0].timestamp,  # Already sorted by timestamp desc
            'latest_score': entries[0].overall_score,
            'latest_passed': entries[0].passed
        }


# Convenience function
def create_audit_logger(
    s3_bucket: Optional[str] = None,
    local_only: bool = False
) -> AuditLogger:
    """
    Create audit logger with sensible defaults.

    Args:
        s3_bucket: S3 bucket name (if None, uses env var AUDIT_LOG_S3_BUCKET)
        local_only: Use only local storage, disable S3

    Returns:
        Configured AuditLogger instance
    """
    if s3_bucket is None:
        s3_bucket = os.environ.get('AUDIT_LOG_S3_BUCKET')

    return AuditLogger(
        s3_bucket=s3_bucket,
        enable_s3=not local_only and s3_bucket is not None
    )
