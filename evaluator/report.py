"""Generate HTML evaluation reports."""

import json
from pathlib import Path
from datetime import datetime


def generate_html_report(results: list, output_path: str = "report.html"):
    """
    Generate a visual HTML report from evaluation results.
    
    Args:
        results: List of evaluation result dictionaries
        output_path:  Where to save the HTML file
    """
    
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Evaluation Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 40px 20px;
            color: #fff;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { 
            text-align: center; 
            margin-bottom: 40px;
            font-size: 2.5rem;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card h3 { font-size: 0.9rem; opacity: 0.7; margin-bottom: 8px; }
        .card .value { font-size: 2.5rem; font-weight: bold; }
        .card.pass .value { color: #00ff88; }
        .card.fail .value { color: #ff4757; }
        .card.neutral .value { color: #00d4ff; }
        
        .evaluation {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 30px;
            margin-bottom:  30px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .evaluation h2 {
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .status-badge {
            padding: 4px 12px;
            border-radius:  20px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        .status-badge.pass { background: #00ff88; color: #000; }
        .status-badge.fail { background: #ff4757; color: #fff; }
        
        .metrics-grid {
            display: grid;
            grid-template-columns:  repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
        }
        .metric {
            background: rgba(0,0,0,0.2);
            border-radius: 12px;
            padding: 20px;
        }
        .metric h4 { 
            font-size: 0.85rem; 
            opacity: 0.7; 
            margin-bottom:  12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .score-bar {
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
            margin:  10px 0;
        }
        .score-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        .score-fill.good { background: linear-gradient(90deg, #00ff88, #00d4ff); }
        .score-fill.medium { background: linear-gradient(90deg, #ffd93d, #ff9f43); }
        .score-fill.bad { background: linear-gradient(90deg, #ff4757, #ff6b81); }
        
        .score-value {
            font-size: 1.8rem;
            font-weight:  bold;
        }
        
        .claims-list {
            margin-top: 15px;
            padding:  15px;
            background:  rgba(255,0,0,0.1);
            border-radius: 8px;
            border-left: 3px solid #ff4757;
        }
        .claims-list h5 { color: #ff4757; margin-bottom: 10px; }
        .claims-list ul { padding-left: 20px; }
        .claims-list li { margin:  5px 0; opacity: 0.9; font-size: 0.9rem; }
        
        .timestamp {
            text-align: center;
            opacity: 0.5;
            margin-top: 40px;
            font-size: 0.85rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>LLM Evaluation Report</h1>
        
        <div class="summary-cards">
            <div class="card neutral">
                <h3>Total Evaluations</h3>
                <div class="value">TOTAL_COUNT</div>
            </div>
            <div class="card pass">
                <h3>Passed</h3>
                <div class="value">PASS_COUNT</div>
            </div>
            <div class="card fail">
                <h3>Failed</h3>
                <div class="value">FAIL_COUNT</div>
            </div>
            <div class="card neutral">
                <h3>Avg Score</h3>
                <div class="value">AVG_SCORE</div>
            </div>
        </div>
        
        EVALUATIONS_HTML
        
        <p class="timestamp">Generated on TIMESTAMP</p>
    </div>
</body>
</html>
    """
    
    # Calculate summary stats
    total = len(results)
    passed = sum(1 for r in results if r.get('passed', False))
    failed = total - passed
    avg_score = sum(r.get('overall_score', 0) for r in results) / total if total > 0 else 0
    
    # Generate evaluation HTML
    evaluations_html = ""
    for i, result in enumerate(results, 1):
        overall = result.get('overall_score', 0)
        is_passed = result.get('passed', False)
        relevance = result.get('relevance', {})
        hallucination = result.get('hallucination', {})
        completeness = result.get('completeness', {})
        latency = result.get('latency', {})
        
        def get_score_class(score, inverse=False):
            if inverse: 
                score = 1 - score
            if score >= 0.7:
                return 'good'
            elif score >= 0.4:
                return 'medium'
            return 'bad'
        
        # Unsupported claims (hallucination details)
        unsupported_html = ""
        unsupported = hallucination.get('unsupported_claims', [])
        if unsupported:
            claims_li = "".join(f"<li>{claim}</li>" for claim in unsupported[:5])
            unsupported_html = f"""
            <div class="claims-list">
                <h5>Unsupported Claims Detected</h5>
                <ul>{claims_li}</ul>
            </div>
            """

        # Completeness analysis (covered vs missing aspects)
        completeness_details_html = ""
        covered_aspects = completeness.get("covered_aspects", []) or []
        missing_aspects = completeness.get("missing_aspects", []) or []

        if covered_aspects or missing_aspects:
            coverage_items = []
            if covered_aspects:
                coverage_items.append(
                    "<li><strong>Covered aspects:</strong> " + ", ".join(covered_aspects) + "</li>"
                )
            if missing_aspects:
                coverage_items.append(
                    "<li><strong>Missing aspects:</strong> " + ", ".join(missing_aspects) + "</li>"
                )

            completeness_details_html = f"""
            <div class="claims-list" style="background: rgba(0, 212, 255, 0.08); margin-top: 20px;">
                <h5>Completeness Analysis</h5>
                <ul>
                    {''.join(coverage_items)}
                </ul>
            </div>
            """

        # High-level explanation of why this evaluation passed/failed
        reason_items = []

        if not relevance.get("is_relevant", True):
            reason_items.append(
                "<li>Low semantic relevance to the user's question and/or retrieved context.</li>"
            )
        if hallucination.get("is_hallucinated", False):
            reason_items.append(
                "<li>Contains one or more unsupported or hallucinated claims.</li>"
            )
        if not completeness.get("is_complete", True):
            reason_items.append(
                "<li>Does not fully cover all key aspects of the user's question.</li>"
            )

        if not reason_items and is_passed:
            reason_items.append(
                "<li>Meets thresholds for relevance, factual accuracy, and completeness.</li>"
            )

        diagnostic_html = ""
        if reason_items:
            heading = "Why this evaluation failed" if not is_passed else "Why this evaluation passed"
            diagnostic_html = f"""
            <div class="claims-list" style="margin-top: 20px;">
                <h5>{heading}</h5>
                <ul>
                    {''.join(reason_items)}
                </ul>
            </div>
            """
        
        evaluations_html += f"""
        <div class="evaluation">
            <h2>
                Evaluation #{i}
                <span class="status-badge {'pass' if is_passed else 'fail'}">
                    {'PASSED' if is_passed else 'FAILED'}
                </span>
            </h2>
            
            <div class="metrics-grid">
                <div class="metric">
                    <h4>Overall Score</h4>
                    <div class="score-value">{overall:.1%}</div>
                    <div class="score-bar">
                        <div class="score-fill {get_score_class(overall)}" style="width: {overall*100}%"></div>
                    </div>
                </div>
                
                <div class="metric">
                    <h4>Relevance</h4>
                    <div class="score-value">{relevance.get('score', 0):.1%}</div>
                    <div class="score-bar">
                        <div class="score-fill {get_score_class(relevance.get('score', 0))}" style="width: {relevance.get('score', 0)*100}%"></div>
                    </div>
                </div>
                
                <div class="metric">
                    <h4>Factual Accuracy (1 - Hallucination)</h4>
                    <div class="score-value">{(1 - hallucination.get('score', 0)):.1%}</div>
                    <div class="score-bar">
                        <div class="score-fill {get_score_class(hallucination.get('score', 0), inverse=True)}" style="width: {(1 - hallucination.get('score', 0))*100}%"></div>
                    </div>
                    {unsupported_html}
                </div>
                
                <div class="metric">
                    <h4>Completeness</h4>
                    <div class="score-value">{completeness.get('score', 0):.1%}</div>
                    <div class="score-bar">
                        <div class="score-fill {get_score_class(completeness.get('score', 0))}" style="width: {completeness.get('score', 0)*100}%"></div>
                    </div>
                </div>
                
                <div class="metric">
                    <h4>Latency</h4>
                    <div class="score-value">{latency.get('total_ms', 0):.0f}ms</div>
                    <div class="score-bar">
                        <div class="score-fill good" style="width: 100%"></div>
                    </div>
                </div>
            </div>

            {completeness_details_html}
            {diagnostic_html}
        </div>
        """
    
    # Replace placeholders
    html = html.replace('TOTAL_COUNT', str(total))
    html = html.replace('PASS_COUNT', str(passed))
    html = html.replace('FAIL_COUNT', str(failed))
    html = html.replace('AVG_SCORE', f"{avg_score:.0%}")
    html = html.replace('EVALUATIONS_HTML', evaluations_html)
    html = html.replace('TIMESTAMP', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # Write file
    Path(output_path).write_text(html)
    print(f"Report generated: {output_path}")
    return output_path