"""Command-line interface for LocalMod."""

import argparse
import sys
import json


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LocalMod - Fully offline content moderation API"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Server command
    server_parser = subparsers.add_parser("serve", help="Start the API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    server_parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze text from command line")
    analyze_parser.add_argument("text", help="Text to analyze")
    analyze_parser.add_argument(
        "--classifiers", "-c",
        nargs="+",
        default=None,
        help="Classifiers to run",
    )
    analyze_parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON",
    )
    
    # Download models command
    download_parser = subparsers.add_parser("download", help="Pre-download models")
    download_parser.add_argument(
        "--classifiers", "-c",
        nargs="+",
        default=None,
        help="Specific classifiers to download",
    )
    
    # List classifiers command
    subparsers.add_parser("list", help="List available classifiers")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        import uvicorn
        uvicorn.run(
            "localmod.api.app:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload,
        )
    
    elif args.command == "analyze":
        from localmod.pipeline import SafetyPipeline
        
        pipeline = SafetyPipeline()
        report = pipeline.analyze(args.text, classifiers=args.classifiers, include_explanation=True)
        
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"\n{'='*50}")
            print(f"Flagged: {report.flagged}")
            print(f"Severity: {report.severity.value}")
            print(f"Summary: {report.summary}")
            print(f"Processing time: {report.processing_time_ms:.2f}ms")
            print(f"{'='*50}")
            
            for result in report.results:
                status = "❌ FLAGGED" if result.flagged else "✓ PASSED"
                print(f"\n{result.classifier}: {status}")
                print(f"  Confidence: {result.confidence:.2%}")
                if result.categories:
                    print(f"  Categories: {', '.join(result.categories)}")
                if result.explanation:
                    print(f"  {result.explanation}")
    
    elif args.command == "download":
        from localmod.pipeline import SafetyPipeline
        from localmod.classifiers import CLASSIFIER_REGISTRY
        
        classifiers = args.classifiers or list(CLASSIFIER_REGISTRY.keys())
        print(f"Downloading models for: {', '.join(classifiers)}")
        
        pipeline = SafetyPipeline(classifiers=classifiers)
        pipeline.load_all()
        
        print("All models downloaded successfully!")
    
    elif args.command == "list":
        from localmod.classifiers import CLASSIFIER_REGISTRY
        
        print("\nAvailable Classifiers:")
        print("=" * 60)
        for name, cls in CLASSIFIER_REGISTRY.items():
            doc = cls.__doc__.strip().split('\n')[0] if cls.__doc__ else "No description"
            print(f"\n  {name}")
            print(f"    Version: {cls.version}")
            print(f"    {doc}")
        print()
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

