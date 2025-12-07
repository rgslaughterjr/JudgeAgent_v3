"""
AWS CDK Stack for Judge Agent v3

Deploys:
- Lambda function for evaluation API
- API Gateway REST API
- S3 bucket for audit logs
- Secrets Manager for credentials
- IAM roles with least-privilege access
- CloudWatch dashboards and alarms
"""

import os
from constructs import Construct
from aws_cdk import (
    App,
    Stack,
    Duration,
    RemovalPolicy,
    CfnOutput,
    Tags,
    aws_lambda as lambda_,
    aws_apigateway as apigw,
    aws_s3 as s3,
    aws_iam as iam,
    aws_secretsmanager as secretsmanager,
    aws_logs as logs,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cw_actions,
    aws_sns as sns,
)


class JudgeAgentStack(Stack):
    """Main infrastructure stack for Judge Agent"""
    
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        # Environment configuration
        env_name = self.node.try_get_context("env") or "dev"
        
        # ====================================================================
        # S3 BUCKET FOR AUDIT LOGS
        # ====================================================================
        
        self.audit_bucket = s3.Bucket(
            self, "AuditLogsBucket",
            bucket_name=f"judge-agent-audit-logs-{env_name}-{self.account}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="ArchiveOldLogs",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INTELLIGENT_TIERING,
                            transition_after=Duration.days(30)
                        ),
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(90)
                        )
                    ],
                    expiration=Duration.days(365)  # Keep logs for 1 year
                )
            ]
        )
        
        # ====================================================================
        # SECRETS MANAGER FOR API CREDENTIALS
        # ====================================================================
        
        self.api_secret = secretsmanager.Secret(
            self, "JudgeAgentApiSecret",
            secret_name=f"judge-agent/{env_name}/api-credentials",
            description="API credentials for Judge Agent evaluation service",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                secret_string_template='{"api_key": ""}',
                generate_string_key="api_key",
                exclude_punctuation=True,
                password_length=32
            )
        )
        
        # ====================================================================
        # IAM ROLE FOR LAMBDA
        # ====================================================================
        
        self.lambda_role = iam.Role(
            self, "JudgeAgentLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Execution role for Judge Agent Lambda function"
        )
        
        # CloudWatch Logs permissions
        self.lambda_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AWSLambdaBasicExecutionRole"
            )
        )
        
        # Bedrock permissions
        self.lambda_role.add_to_policy(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
            ],
            resources=[
                f"arn:aws:bedrock:*:{self.account}:inference-profile/*",
                "arn:aws:bedrock:*::foundation-model/*"
            ]
        ))
        
        # S3 audit bucket permissions
        self.audit_bucket.grant_read_write(self.lambda_role)
        
        # Secrets Manager permissions
        self.api_secret.grant_read(self.lambda_role)
        
        # ====================================================================
        # LAMBDA FUNCTION
        # ====================================================================
        
        self.lambda_function = lambda_.Function(
            self, "JudgeAgentFunction",
            function_name=f"judge-agent-{env_name}",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="lambda_handler.handler",
            code=lambda_.Code.from_asset("../lambda"),
            role=self.lambda_role,
            timeout=Duration.minutes(5),
            memory_size=1024,
            environment={
                "ENV": env_name,
                "AUDIT_BUCKET": self.audit_bucket.bucket_name,
                "SECRET_ARN": self.api_secret.secret_arn,
                "CLAUDE_MODEL_ID": "anthropic.claude-sonnet-4-20250514-v1:0",
                "LOG_LEVEL": "INFO"
            },
            log_retention=logs.RetentionDays.TWO_WEEKS
        )
        
        # ====================================================================
        # API GATEWAY
        # ====================================================================
        
        self.api = apigw.RestApi(
            self, "JudgeAgentApi",
            rest_api_name=f"Judge Agent API ({env_name})",
            description="REST API for AI Agent evaluation",
            deploy_options=apigw.StageOptions(
                stage_name=env_name,
                throttling_rate_limit=100,
                throttling_burst_limit=200,
                logging_level=apigw.MethodLoggingLevel.INFO,
                data_trace_enabled=True,
                metrics_enabled=True
            ),
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=["GET", "POST", "OPTIONS"],
                allow_headers=["Content-Type", "Authorization", "X-Api-Key"]
            )
        )
        
        # Lambda integration
        lambda_integration = apigw.LambdaIntegration(
            self.lambda_function,
            request_templates={"application/json": '{"statusCode": "200"}'}
        )
        
        # API resources
        evaluate = self.api.root.add_resource("evaluate")
        evaluate.add_method("POST", lambda_integration)
        
        health = self.api.root.add_resource("health")
        health.add_method("GET", lambda_integration)
        
        results = self.api.root.add_resource("results")
        results.add_method("GET", lambda_integration)
        
        results_by_agent = results.add_resource("{agent_id}")
        results_by_agent.add_method("GET", lambda_integration)
        
        # ====================================================================
        # CLOUDWATCH DASHBOARD
        # ====================================================================
        
        self.dashboard = cloudwatch.Dashboard(
            self, "JudgeAgentDashboard",
            dashboard_name=f"JudgeAgent-{env_name}"
        )
        
        # Lambda metrics
        self.dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Lambda Invocations & Errors",
                left=[
                    self.lambda_function.metric_invocations(),
                    self.lambda_function.metric_errors()
                ],
                width=12
            ),
            cloudwatch.GraphWidget(
                title="Lambda Duration",
                left=[self.lambda_function.metric_duration()],
                width=12
            )
        )
        
        # API Gateway metrics
        self.dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="API Requests",
                left=[
                    self.api.metric_count(),
                    self.api.metric_client_error(),
                    self.api.metric_server_error()
                ],
                width=12
            ),
            cloudwatch.GraphWidget(
                title="API Latency",
                left=[self.api.metric_latency()],
                width=12
            )
        )
        
        # ====================================================================
        # CLOUDWATCH ALARMS
        # ====================================================================
        
        # SNS topic for alarms
        self.alarm_topic = sns.Topic(
            self, "JudgeAgentAlarmTopic",
            topic_name=f"judge-agent-alarms-{env_name}"
        )
        
        # Lambda error alarm
        error_alarm = cloudwatch.Alarm(
            self, "LambdaErrorAlarm",
            metric=self.lambda_function.metric_errors(),
            threshold=5,
            evaluation_periods=1,
            alarm_description="Judge Agent Lambda errors exceeded threshold",
            alarm_name=f"judge-agent-{env_name}-lambda-errors"
        )
        error_alarm.add_alarm_action(cw_actions.SnsAction(self.alarm_topic))
        
        # Lambda duration alarm
        duration_alarm = cloudwatch.Alarm(
            self, "LambdaDurationAlarm",
            metric=self.lambda_function.metric_duration(),
            threshold=240000,  # 4 minutes (80% of 5-min timeout)
            evaluation_periods=1,
            alarm_description="Judge Agent Lambda duration exceeds 4 minutes",
            alarm_name=f"judge-agent-{env_name}-lambda-duration"
        )
        duration_alarm.add_alarm_action(cw_actions.SnsAction(self.alarm_topic))
        
        # API 5XX alarm
        api_error_alarm = cloudwatch.Alarm(
            self, "Api5xxAlarm",
            metric=self.api.metric_server_error(),
            threshold=10,
            evaluation_periods=1,
            alarm_description="Judge Agent API 5XX errors exceeded threshold",
            alarm_name=f"judge-agent-{env_name}-api-5xx"
        )
        api_error_alarm.add_alarm_action(cw_actions.SnsAction(self.alarm_topic))
        
        # ====================================================================
        # OUTPUTS
        # ====================================================================
        
        CfnOutput(
            self, "ApiEndpoint",
            value=self.api.url,
            description="API Gateway endpoint URL"
        )
        
        CfnOutput(
            self, "AuditBucketName",
            value=self.audit_bucket.bucket_name,
            description="S3 bucket for audit logs"
        )
        
        CfnOutput(
            self, "LambdaFunctionArn",
            value=self.lambda_function.function_arn,
            description="Lambda function ARN"
        )
        
        CfnOutput(
            self, "SecretArn",
            value=self.api_secret.secret_arn,
            description="Secrets Manager secret ARN"
        )
        
        CfnOutput(
            self, "DashboardUrl",
            value=f"https://{self.region}.console.aws.amazon.com/cloudwatch/home?region={self.region}#dashboards:name={self.dashboard.dashboard_name}",
            description="CloudWatch dashboard URL"
        )
        
        # ====================================================================
        # TAGS
        # ====================================================================
        
        Tags.of(self).add("Project", "JudgeAgent")
        Tags.of(self).add("Environment", env_name)
        Tags.of(self).add("ManagedBy", "CDK")


# ============================================================================
# APP ENTRY POINT
# ============================================================================

app = App()

# Dev stack
JudgeAgentStack(
    app, "JudgeAgentDev",
    env={
        "account": os.environ.get("CDK_DEFAULT_ACCOUNT"),
        "region": os.environ.get("CDK_DEFAULT_REGION", "us-east-1")
    }
)

# Production stack (optional)
# JudgeAgentStack(
#     app, "JudgeAgentProd",
#     env={"account": "PROD_ACCOUNT_ID", "region": "us-east-1"}
# )

app.synth()
