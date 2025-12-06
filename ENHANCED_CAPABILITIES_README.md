# Judge Agent - Enhanced Dimension Capabilities

## Capability Assessment Summary

| Dimension | Previous State | Enhanced State |
|-----------|---------------|----------------|
| **Performance** | Basic enum only | Full latency measurement, throughput, behavioral analysis |
| **User Experience** | Generic prompts | Specialized UX evaluation: clarity, helpfulness, tone, error handling, guidance |
| **Bias Detection** | Listed, minimal | Paired demographic tests, category-level scoring, stereotype detection |
| **Harm Prevention** | Basic blocking tests | **LLM vs Agent control attribution**, bypass detection, category tracking |
| **Guardrail Effectiveness** | Mentioned only | Boundary testing, bypass resistance, edge case handling, type-specific scores |

---

## New Capabilities Added

### 1. Performance & Behavior (`PerformanceEvaluator`)

**What's New:**
- **Actual latency measurement** (p50, p95, p99 percentiles)
- **Throughput calculation** (requests per second)
- **Behavioral consistency testing** via semantically equivalent query pairs
- **Error recovery analysis** with LLM-based quality scoring
- **Context retention testing** for multi-turn conversations

**Metrics Returned:**
```python
PerformanceResult(
    metrics=PerformanceMetrics(
        latency_ms=1234,
        latency_p50=1100,
        latency_p95=2500,
        latency_p99=4200,
        throughput_rps=8.5,
        error_rate=0.02
    ),
    behavior=BehaviorAnalysis(
        consistency_score=0.89,
        determinism_score=0.92,
        context_retention=0.85,
        recovery_score=0.78
    )
)
```

---

### 2. User Experience (`UXEvaluator`)

**What's New:**
- **5-factor UX model**: Clarity, Helpfulness, Tone, Error Handling, Guidance
- **Persona-based testing** (technical vs non-technical users)
- **Context-appropriate evaluation** (formal, casual, sensitive scenarios)
- **Accessibility considerations**

**Test Categories:**
- Clarity: Technical explanations, jargon handling
- Helpfulness: Task completion, problem-solving
- Tone: Formal/casual/sensitive context matching
- Error Handling: Limitation communication, alternatives offered
- Guidance: Onboarding, feature discovery

---

### 3. Bias Detection (`BiasEvaluator`)

**What's New:**
- **Paired demographic testing**: Same prompt, only demographic indicator changes
- **6 demographic categories**: Gender, Race, Age, Disability, Socioeconomic, Nationality
- **Bias type classification**: Favoritism, Stereotyping, Exclusion, Condescension
- **Quality differential measurement** between demographic groups

**Key Innovation:**
```python
BiasTestResult(
    category="gender",
    group_a="male applicant",
    group_b="female applicant",
    quality_differential=0.15,  # 15% quality gap = bias detected
    detected_bias=True,
    bias_type="favoritism"
)
```

---

### 4. Harm Prevention (`HarmPreventionEvaluator`)

**What's New - CRITICAL DISTINCTION:**

The evaluator now **differentiates between control types**:

| Control Type | Description | Example |
|-------------|-------------|---------|
| `llm_builtin` | Generic LLM safety (Claude's training) | Refuses "how to make a bomb" |
| `agent_specific` | Developer-added safety logic | Domain-specific refusals, context-aware blocking |
| `guardrail_service` | External service (Bedrock Guardrails) | Policy-based filtering |

**Why This Matters:**
- If `llm_catches >> agent_catches`: Agent relies too heavily on LLM defaults
- If `agent_catches` is high: Good domain-specific safety engineering
- If `neither_catches > 0`: **CRITICAL** - harmful requests getting through

**Metrics Returned:**
```python
HarmPreventionResult(
    llm_control_score=0.45,      # 45% caught by LLM
    agent_control_score=0.35,    # 35% caught by agent logic
    guardrail_service_score=0.15,# 15% caught by external guardrail
    neither_catches=2,           # 2 harmful requests NOT blocked!
    
    # Control attribution
    llm_catches=18,
    agent_catches=14,
    both_catches=6,
    neither_catches=2
)
```

**Bypass Techniques Tested:**
- Hypothetical framing ("imagine if...")
- Roleplay ("pretend you're a...")
- Academic context ("for research purposes...")
- Multi-turn escalation
- Encoding tricks

---

### 5. Guardrail Effectiveness (`GuardrailEvaluator`)

**What's New:**
- **Guardrail type-specific scoring**: Topic, PII, Content, Word, Custom
- **Boundary consistency testing**: Same boundary, different phrasings
- **Edge case handling**: Near-boundary inputs
- **Bypass resistance scoring**: Resistance to obfuscation, encoding, roleplay

**Key Metrics:**
```python
GuardrailResult(
    topic_blocking_score=0.92,
    pii_filtering_score=0.88,
    content_filtering_score=0.95,
    boundary_consistency=0.90,
    edge_case_handling=0.85,
    bypass_resistance=0.82,
    bypass_vulnerabilities=["l33t speak bypass on word filter"]
)
```

---

## Integration with Existing Code

### Import into `judge_agent_langchain.py`:

```python
from judge_agent_enhanced_dimensions import (
    EnhancedDimensionEvaluator,
    PerformanceResult,
    UXResult,
    BiasResult,
    HarmPreventionResult,
    GuardrailResult
)

# Use unified evaluator
evaluator = EnhancedDimensionEvaluator()

# Run all enhanced dimensions
results = await evaluator.evaluate_all(
    connector=connector,
    agent_context=agent_context_string,
    test_counts={"harm": 50, "bias": 40}  # Optional custom counts
)

# Or single dimension
harm_result = await evaluator.evaluate_dimension(
    "harm_prevention", 
    connector, 
    agent_context, 
    num_tests=30
)
```

### Add to LangGraph Supervisor:

```python
# In judge_agent_supervisor.py, add enhanced evaluator nodes:

async def enhanced_harm_node(state: SupervisorState) -> dict:
    evaluator = HarmPreventionEvaluator(get_llm())
    result = await evaluator.evaluate(connector, build_context(state))
    return {"harm_result": result}
```

---

## Test Generation Quality

Each evaluator generates **specialized prompts** for its dimension:

| Dimension | Test Focus | Example Prompts |
|-----------|------------|-----------------|
| Performance | Latency, consistency | Simple vs complex queries, equivalent phrasings |
| UX | Clarity, tone | Technical explanations, error scenarios |
| Bias | Demographic parity | "Help John with investment" vs "Help Maria with investment" |
| Harm | Control bypass | Direct, hypothetical, roleplay, multi-turn attacks |
| Guardrails | Boundary probing | Edge cases, obfuscation, escalation |

---

## Files Delivered

1. **`judge_agent_enhanced_dimensions.py`** - Complete enhanced evaluators
2. **`judge_agent_langchain.py`** - Core LangChain implementation (updated)
3. **`judge_agent_supervisor.py`** - Parallel supervisor architecture
4. **`requirements_langchain.txt`** - Dependencies

---

## Key Differentiators

✅ **LLM vs Agent control attribution** - Know exactly where your safety comes from  
✅ **Paired demographic testing** - Rigorous bias detection methodology  
✅ **Actual latency measurement** - Real performance metrics, not estimates  
✅ **Bypass resistance scoring** - Know if your guardrails can be circumvented  
✅ **UX factor model** - Actionable UX improvement recommendations
