"""
AWS Lambda Handler for Judge Agent v3

Exposes REST API endpoints:
- POST /evaluate - Evaluate an AI agent
- GET /health - Health check
- GET /results - List evaluation results
- GET /results/{agent_id} - Get results for specific agent
"""

import json
import os
import logging
import traceback
from datetime import datetime
from typing import Any

import boto3

# Configure logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


def get_secret(secret_arn: str) -> dict:
    """Retrieve secret from Secrets Manager"""
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=secret_arn)
    return json.loads(response["SecretString"])


def validate_api_key(event: dict) -> bool:
    """Validate API key from request headers"""
    try:
        secret_arn = os.environ.get("SECRET_ARN")
        if not secret_arn:
            logger.warning("SECRET_ARN not configured - skipping auth")
            return True
        
        headers = event.get("headers", {})
        api_key = headers.get("x-api-key") or headers.get("X-Api-Key")
        
        if not api_key:
            return False
        
        secret = get_secret(secret_arn)
        return api_key == secret.get("api_key")
    except Exception as e:
        logger.error(f"API key validation error: {e}")
        return False


def create_response(status_code: int, body: Any) -> dict:
    """Create API Gateway response"""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization,X-Api-Key"
        },
        "body": json.dumps(body, default=str)
    }


def handle_health(event: dict) -> dict:
    """Health check endpoint"""
    return create_response(200, {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "3.0.0",
        "environment": os.environ.get("ENV", "unknown")
    })


async def handle_evaluate(event: dict) -> dict:
    """
    Evaluate an AI agent.
    
    Request body:
    {
        "agent_id": "string",
        "agent_name": "string",
        "framework": "langchain|crewai|autogen|...",
        "description": "string",
        "risk_level": "low|medium|high|critical",
        "data_access": ["list", "of", "data", "sources"],
        "dimensions": ["optional", "list", "of", "dimensions"],
        "evaluator_user": "user@email.com"
    }
    """
    try:
        body = json.loads(event.get("body", "{}"))
        
        # Validate required fields
        required_fields = ["agent_id", "agent_name", "framework"]
        missing = [f for f in required_fields if f not in body]
        if missing:
            return create_response(400, {
                "error": "Missing required fields",
                "missing": missing
            })
        
        # Import evaluation module (lazy import for cold start optimization)
        from judge_agent_supervisor import (
            JudgeAgentSupervisor,
            AgentConfig,
            MockAgent
        )
        
        # Create agent config
        config = AgentConfig(
            agent_id=body["agent_id"],
            name=body["agent_name"],
            framework=body["framework"],
            risk_level=body.get("risk_level", "medium"),
            description=body.get("description", ""),
            data_access=body.get("data_access", [])
        )
        
        # For now, use MockAgent - in production, this would connect to actual agent
        # TODO: Implement real agent connectors based on framework
        connector = MockAgent()
        judge = JudgeAgentSupervisor(connector)
        
        # Run evaluation
        evaluator_user = body.get("evaluator_user", "api-user")
        result = await judge.evaluate_parallel(config, evaluator_user)
        
        return create_response(200, {
            "success": True,
            "evaluation_id": result.get("evaluation_id"),
            "agent_id": body["agent_id"],
            "overall_score": result["overall_score"],
            "passes_gate": result["passes_gate"],
            "dimension_results": result["dimension_results"],
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except json.JSONDecodeError:
        return create_response(400, {"error": "Invalid JSON body"})
    except Exception as e:
        logger.error(f"Evaluation error: {traceback.format_exc()}")
        return create_response(500, {
            "error": "Evaluation failed",
            "message": str(e)
        })


def handle_results(event: dict) -> dict:
    """List evaluation results"""
    try:
        # Get query parameters
        params = event.get("queryStringParameters") or {}
        agent_id = params.get("agent_id")
        limit = int(params.get("limit", 50))
        
        # Import audit logger
        from utils import create_audit_logger
        
        audit_logger = create_audit_logger(
            s3_bucket=os.environ.get("AUDIT_BUCKET"),
            local_only=not os.environ.get("AUDIT_BUCKET")
        )
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            entries = loop.run_until_complete(
                audit_logger.query_history(agent_id=agent_id)
            )
            
            # Limit results
            entries = entries[:limit]
            
            return create_response(200, {
                "count": len(entries),
                "results": [e.model_dump() for e in entries]
            })
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Results query error: {traceback.format_exc()}")
        return create_response(500, {
            "error": "Query failed",
            "message": str(e)
        })


def handle_results_by_agent(event: dict, agent_id: str) -> dict:
    """Get evaluation results for specific agent"""
    try:
        params = event.get("queryStringParameters") or {}
        days = int(params.get("days", 30))
        
        from utils import create_audit_logger
        
        audit_logger = create_audit_logger(
            s3_bucket=os.environ.get("AUDIT_BUCKET"),
            local_only=not os.environ.get("AUDIT_BUCKET")
        )
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            stats = loop.run_until_complete(
                audit_logger.get_agent_statistics(agent_id, days=days)
            )
            
            return create_response(200, stats)
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Agent stats error: {traceback.format_exc()}")
        return create_response(500, {
            "error": "Stats query failed",
            "message": str(e)
        })


def handler(event: dict, context: Any) -> dict:
    """
    Main Lambda handler - routes requests to appropriate handlers.
    """
    logger.info(f"Received event: {json.dumps(event, default=str)[:500]}")
    
    # Get HTTP method and path
    http_method = event.get("httpMethod", "GET")
    path = event.get("path", "/")
    path_params = event.get("pathParameters") or {}
    
    # Health check (no auth required)
    if path == "/health":
        return handle_health(event)
    
    # Validate API key for other endpoints
    if not validate_api_key(event):
        return create_response(401, {"error": "Unauthorized - invalid or missing API key"})
    
    # Route to appropriate handler
    try:
        if path == "/evaluate" and http_method == "POST":
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(handle_evaluate(event))
            finally:
                loop.close()
        
        elif path == "/results" and http_method == "GET":
            return handle_results(event)
        
        elif path.startswith("/results/") and http_method == "GET":
            agent_id = path_params.get("agent_id") or path.split("/")[-1]
            return handle_results_by_agent(event, agent_id)
        
        else:
            return create_response(404, {
                "error": "Not found",
                "path": path,
                "method": http_method
            })
            
    except Exception as e:
        logger.error(f"Handler error: {traceback.format_exc()}")
        return create_response(500, {
            "error": "Internal server error",
            "message": str(e)
        })
