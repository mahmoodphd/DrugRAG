#!/usr/bin/env python3
"""
Drug Repurposing LLM Pipeline - CLI Entry Point.

Usage:
    python run.py info                             # Print config and dataset stats
    python run.py phase1 [options]                 # LLM baseline benchmarking
    python run.py phase2 [options]                 # Literature corpus construction
    python run.py phase3 [options]                 # LightRAG knowledge graph
    python run.py phase4 [options]                 # Multi-LLM evaluation
"""

import argparse
import sys


def cmd_info(args):
    """Print configuration summary and dataset statistics."""
    import config
    import data_loader

    config.print_config_summary()

    df = data_loader.load_raw_dataset()
    df = data_loader.clean_text_columns(df)
    unique_drugs = data_loader.extract_unique_drugs(df)
    data_loader.print_dataset_stats(unique_drugs)

    # Test API connectivity
    print("\n  Testing API connectivity...")
    try:
        client = config.get_openrouter_client()
        resp = client.chat.completions.create(
            model=config.DEFAULT_MODEL,
            messages=[{"role": "user", "content": "Say 'OK'"}],
            max_tokens=5,
        )
        print(f"  OpenRouter: CONNECTED ({resp.choices[0].message.content})")
    except Exception as e:
        print(f"  OpenRouter: FAILED ({str(e)[:60]})")


def cmd_phase1(args):
    """Run Phase 1: LLM Baseline Benchmarking."""
    import phase1_baseline

    modes = [args.mode] if args.mode != "all" else None
    phase1_baseline.run_baseline(
        subset_name=args.subset,
        modes=modes,
        model=args.model,
        limit=args.limit,
    )


def cmd_phase2(args):
    """Run Phase 2: Literature Corpus Construction."""
    import phase2_corpus

    phase2_corpus.build_corpus(
        subset_name="all",
        limit=args.limit,
        sources=args.source,
    )


def cmd_phase3(args):
    """Run Phase 3: LightRAG Knowledge Graph."""
    import phase3_knowledge_graph as kg

    if args.ingest:
        if args.priority:
            priority_files = kg.compute_ingestion_priority()
            kg.ingest_corpus(limit=args.limit, priority_files=priority_files)
        else:
            kg.ingest_corpus(limit=args.limit)
    elif args.query:
        mode = args.mode or "hybrid"
        result = kg.query_kg(args.query, mode=mode)
        print(f"\nMode: {mode}")
        print(f"Response:\n{result['response']}")
    elif args.batch_query:
        kg.batch_query(questions_file=args.batch_query)
    elif args.stats:
        stats = kg.export_graph_stats()
        import json
        print(json.dumps(stats, indent=2))
    else:
        print("Phase 3 options:")
        print("  --ingest [--limit N] [--priority]  Ingest corpus into knowledge graph")
        print("  --query TEXT [--mode M]            Query the knowledge graph")
        print("  --batch-query FILE                 Batch query from JSONL file")
        print("  --stats                            Export graph statistics")


def cmd_phase4(args):
    """Run Phase 4: Multi-LLM Evaluation."""
    import phase4_evaluation

    models = args.models.split(",") if args.models else None
    configs = args.configs.split(",") if args.configs else None

    # Run evaluation
    results = phase4_evaluation.run_evaluation(
        models=models,
        questions_file=args.questions,
        configs=configs,
    )

    # Generate report
    phase4_evaluation.generate_comparison_report(results)

    # Optional RAGAS evaluation
    if args.ragas:
        phase4_evaluation.run_ragas_evaluation(
            results,
            ground_truth_file=args.ground_truth,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Drug Repurposing LLM Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline phase")

    # --- info ---
    subparsers.add_parser("info", help="Print configuration and dataset stats")

    # --- phase1 ---
    p1 = subparsers.add_parser("phase1", help="LLM Baseline Benchmarking")
    p1.add_argument("--subset", default="pilot30",
                     choices=["pilot30", "pc4", "pc7", "pc2", "all"],
                     help="Drug subset to query (default: pilot30)")
    p1.add_argument("--mode", default="all",
                     choices=["parametric", "standard_search", "deep_search", "all"],
                     help="Query mode (default: all)")
    p1.add_argument("--model", default=None,
                     help="OpenRouter model ID (default: from config)")
    p1.add_argument("--limit", type=int, default=None,
                     help="Max drugs to process")

    # --- phase2 ---
    p2 = subparsers.add_parser("phase2", help="Literature Corpus Construction")
    p2.add_argument("--limit", type=int, default=None,
                     help="Max drugs to search")
    p2.add_argument("--source", default="both",
                     choices=["pubmed", "scholar", "both"],
                     help="Search source (default: both)")

    # --- phase3 ---
    p3 = subparsers.add_parser("phase3", help="LightRAG Knowledge Graph")
    p3.add_argument("--ingest", action="store_true",
                     help="Ingest corpus into knowledge graph")
    p3.add_argument("--query", type=str, default=None,
                     help="Query the knowledge graph")
    p3.add_argument("--batch-query", type=str, default=None,
                     help="Batch query from JSONL file")
    p3.add_argument("--stats", action="store_true",
                     help="Export graph statistics")
    p3.add_argument("--mode", default="hybrid",
                     choices=["local", "global", "hybrid"],
                     help="Query mode (default: hybrid)")
    p3.add_argument("--limit", type=int, default=None,
                     help="Max files to ingest")
    p3.add_argument("--priority", action="store_true",
                     help="Ingest papers by priority ranking (eval drugs first)")

    # --- phase4 ---
    p4 = subparsers.add_parser("phase4", help="Multi-LLM Evaluation")
    p4.add_argument("--models", default=None,
                     help="Comma-separated model IDs (default: from config)")
    p4.add_argument("--questions", default=None,
                     help="Path to evaluation questions file (JSON/JSONL)")
    p4.add_argument("--ragas", action="store_true",
                     help="Run RAGAS evaluation metrics")
    p4.add_argument("--configs", default=None,
                     help="Comma-separated config names (default: all)")
    p4.add_argument("--ground-truth", default=None,
                     help="Path to ground truth JSON for RAGAS evaluation")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "info": cmd_info,
        "phase1": cmd_phase1,
        "phase2": cmd_phase2,
        "phase3": cmd_phase3,
        "phase4": cmd_phase4,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
