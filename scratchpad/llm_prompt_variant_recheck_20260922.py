"""Re-check of the 3 LLM Agent prompt-variant IC/IR comparison, last done
2026-09-03 (only 2 rolling windows then). ~19 days later -- checking
whether MIN_WINDOWS_FOR_IR_TRUST=4 has now been reached and whether the
picture has changed.
"""
import ic_tracking

labels = {
    "balanced": "llm_agent",
    "qualitative_weighted": "llm_agent_qualitative_weighted",
    "evidence_strict": "llm_agent_evidence_strict",
}

for variant, strategy in labels.items():
    report = ic_tracking.methodology_report(strategy)
    weight = ic_tracking.ensemble_weight(report)
    print(f"\n=== {variant} ({strategy}) ===")
    print(f"n_settled={report['n_settled']} effective_n_settled={report['effective_n_settled']} "
          f"trust_floor_met={report['trust_floor_met']}")
    print(f"overall_ic={report['overall_ic']} ir={report['ir']} "
          f"num_windows={len(report['ic_series'])} ensemble_weight={weight}")
    for w in report["ic_series"]:
        print("  window:", w)
