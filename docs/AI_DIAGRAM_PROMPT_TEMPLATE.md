# AI System Diagram Generation Prompt

**Instructions:** Copy and paste the text below into your AI assistant (Claude, GPT, Gemini) to generate professional, error-free architecture diagrams for your Agentic Systems.

---

## 📋 Prompt for AI

**Role:** You are an Expert Technical Documenter and System Architect.

**Objective:** Create a comprehensive "Architecture Overview" for my AI Agent system. You must generate a single, self-contained HTML file named `architecture_diagrams.html` that I can open in any browser.

### 1. Visual Requirements (Mermaid.js)

You will use **Mermaid.js** embedded in the HTML.
**CRITICAL SYNTAX RULES (To prevent "Syntax Error" bugs):**

1. **Strict Quoting:** ALL participant aliases and subgraph labels MUST be double-quoted if they contain spaces or special characters.
    * *Bad:* `subgraph AWS Cloud`
    * *Good:* `subgraph AWS_Cloud ["AWS Cloud (Region A)"]`
    * *Bad:* `participant A as User Service`
    * *Good:* `participant A as "User Service"`
2. **Simple Formatting:** Use `<br/>` for line breaks. Avoid complex HTML tags (like `<b>`, `<i>`) inside the Mermaid labels as they often break the renderer.
3. **Sequence Diagrams:** Do NOT use square brackets `[]` for participant aliases. Use standard quotes `""`.

### 2. Required Diagrams

Please include these 4 specific diagrams in the HTML file:

1. **System Map (Flowchart/Graph)**:
    * Show High-Level infrastructure (AWS Lambda, API Gateway, Docker containers, etc.).
    * Group components using specific Subgraphs (e.g., `subgraph Cloud ["AWS Cloud"]`).
    * Show how the User connects to the System.

2. **Sequence & Cost Flow (Sequence Diagram)**:
    * Show a step-by-step request lifecycle.
    * **Crucial:** highlighting exactly *where* money/tokens are spent (e.g., "Note over LLM: $$$ Cost Here").

3. **Internal Decision Logic (Flowchart)**:
    * Show the internal "Brain" of the Agent (e.g., LangGraph state machine, Router logic).
    * Visualize parallel execution or conditional paths (If X -> do Y).

4. **The AI Intelligence Flow (Sequence Diagram)**:
    * Focus specifically on the **"AI Moment"**.
    * Show the exact Prompt being sent to the LLM.
    * Show the "Reasoning" or "Thinking" phase.
    * Show the Structured Output returned (JSON).

### 3. HTML Skeleton (Copy This Format)

Use this exact HTML structure for the output file:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Architecture</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({ startOnLoad: true, theme: 'default', securityLevel: 'loose' });
    </script>
    <style>
        body { font-family: sans-serif; background: #f4f6f8; padding: 20px; max-width: 1200px; margin: 0 auto; }
        .container { background: white; padding: 40px; margin-bottom: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: #333; }
        h2 { color: #444; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        .mermaid { text-align: center; margin: 30px 0; }
    </style>
</head>
<body>
    <h1>System Architecture</h1>
    
    <!-- INSERT DIAGRAMS HERE INSIDE CONTAINERS -->
    <div class="container">
        <h2>1. System Overview</h2>
        <div class="mermaid">
            graph TD
            User([User]) --> API["API Gateway"]
            %% ... insert graph logic here ...
        </div>
    </div>
    
</body>
</html>
```
