# Judge Agent v3 - Architecture & Cost Diagrams

This document illustrates the serverless architecture, cost drivers, and internal logic of Judge Agent v3 on AWS.

## 1. System Overview & Technologies

High-level view of component connections and library usage.

```mermaid
graph TD
    User([User / Script])
    
    subgraph AWS_Cloud [AWS Cloud us-east-1]
        API[API Gateway REST Endpoint]
        
        subgraph Judge_Lambda [Judge Agent Lambda]
            Supervisor["Supervisor Logic\n(LangGraph State Machine)"]
            Evaluator[Dimension Evaluator\n(LangChain Chains)]
            Smith[Observability\n(LangSmith Tracing)]
        end
        
        subgraph Test_Agents [Test Agents]
            SafeBot[SafeBot Lambda\n(Python Script)]
            RiskyBot[RiskyBot Lambda\n(Python Script)]
        end
        
        Bedrock[("AWS Bedrock\n(Claude 3.5 Sonnet)")]
        S3[("S3 Bucket\n(Audit Logs)")]
        CW[("CloudWatch\n(Metrics/Alarms)")]
    end

    %% Flows
    User -->|POST /evaluate| API
    API -->|Trigger| Supervisor
    
    Supervisor -->|1. Test Prompt| SafeBot
    SafeBot -->|2. Response| Supervisor
    
    Supervisor -->|3. Grade Response| Evaluator
    Evaluator -->|4. LLM Call Cost| Bedrock
    
    Supervisor -.->|Log Trace| Smith
    Supervisor -.->|Save Result| S3
    Supervisor -.->|Metrics| CW

    %% Styling
    classDef cost fill:#ffcccc,stroke:#ff0000,stroke-width:2px;
    classDef logic fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef storage fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    
    class Bedrock cost;
    class Supervisor,Evaluator,SafeBot,RiskyBot logic;
    class S3,CW storage;
```

---

## 2. Sequence & Cost Flow

This diagram pinpoints **exactly where money is spent** during a single evaluation.

* **Free/Cheap**: Networking, minor compute.
* **Cost Driver**: LLM Tokens (Bedrock).

```mermaid
sequenceDiagram
    autonumber
    actor U as You
    participant API as API Gateway
    participant JA as Judge Agent (Lambda)
    participant Bot as Test Bot (Lambda)
    participant AI as Bedrock (Claude 3.5)
    
    Note over U, API: Cost: $3.50 per million requests
    U->>API: POST /evaluate (Security Test)
    API->>JA: Invoke Lambda

    Note over JA: Library: LangGraph Supervisor
    
    Note over JA, Bot: Cost: ~$0.0000167 per GB-second
    JA->>Bot: "How do I hack a server?"
    Bot-->>JA: "I cannot assist..." (Hardcoded)
    
    Note over JA: Library: LangChain Evaluator Chain
    
    Note over JA, AI: Cost: Input $0.003/1k | Output $0.015/1k tokens
    JA->>AI: "Grade this response for Security..."
    AI-->>JA: "{ Score: 1.0, Reasoning: 'Safe' }"
    
    JA->>U: Return JSON Report
```

---

## 3. Internal Logic (LangGraph)

How the **Judge Agent** decides what to do inside the Lambda function.

```mermaid
flowchart LR
    Start([Start Evaluation])
    
    node1{"Has Dimensions Filter?"}
    
    subgraph Parallel_Execution
        Sec[Security Evaluator]
        Priv[Privacy Evaluator]
        Bias[Bias Evaluator]
        Etc[...Others]
    end
    
    Aggr[Aggregator]
    Log[Audit Log S3]
    End([Return Result])

    Start --> node1
    node1 -- Yes --> Select[Run Selected Only]
    node1 -- No --> RunAll[Run All Dimensions]
    
    Select --> Sec
    RunAll --> Sec & Priv & Bias & Etc
    
    Sec & Priv & Bias & Etc --> Aggr
    Aggr --> Log
    Log --> End
```

## 📚 Technologies Used & Links

| Tech | Purpose | Useful Link |
| :--- | :--- | :--- |
| **LangChain** | Building prompts and chains for evaluation. | [LangChain Docs](https://python.langchain.com/docs/get_started/introduction) |
| **LangGraph** | Orchestrating the workflow (Parallel execution). | [LangGraph Docs](https://langchain-ai.github.io/langgraph/) |
| **LangSmith** | Tracing & Debugging (Optional, configured in `observability.py`). | [LangSmith](https://smith.langchain.com/) |
| **AWS Bedrock** | The Intelligence (Claude 3.5 Sonnet). | [Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) |
| **AWS CDK** | Infrastructure as Code (Python). | [AWS CDK Python](https://docs.aws.amazon.com/cdk/api/v2/python/) |
