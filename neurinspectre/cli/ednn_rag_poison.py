#!/usr/bin/env python3
"""
EDNN RAG Poisoning Attack CLI
Standalone command for RAG vector database poisoning
"""

import argparse
from pathlib import Path
from datetime import datetime


def run_ednn_rag_poison(args):
    """Execute EDNN RAG poisoning analysis.

    Notes:
    - This command does NOT connect to external vector DB services.
    - It produces an optimized embedding and metrics you can use to validate retrieval impact
      in your own (authorized) RAG ingestion/retrieval pipeline.
    """
    from neurinspectre.attacks.ednn_attack import EDNNAttack, EDNNConfig, load_embedding_model
    import numpy as np
    import torch

    print("\n🔴 EDNN RAG Poisoning (retrieval-targeted embedding optimization)\n")

    # Resolve device preference (avoid passing 'auto' into torch.device)
    dev_pref = getattr(args, "device", "cpu") or "cpu"
    if dev_pref == "auto":
        if torch.backends.mps.is_available():
            dev_pref = "mps"
        elif torch.cuda.is_available():
            dev_pref = "cuda"
        else:
            dev_pref = "cpu"

    # Load embedding model/tokenizer (required)
    print(f"📦 Loading embedding model: {args.model_path} (device={dev_pref})")
    model, tokenizer = load_embedding_model(args.model_path, device=dev_pref)
    if model is None or tokenizer is None:
        print("❌ Failed to load embedding model/tokenizer.")
        return 1
    print("✅ Embedding model loaded")

    # Load malicious document
    print(f"\n📄 Loading document: {args.malicious_doc}")
    try:
        with open(args.malicious_doc, 'r', encoding='utf-8') as f:
            malicious_content = f.read()
        if not malicious_content.strip():
            raise ValueError("Document is empty")
        print(f"✅ Loaded document ({len(malicious_content)} chars)")
    except Exception as e:
        print(f"❌ Failed to load document: {e}")
        return 1

    # Optional: local vector DB embeddings for rank-based evaluation
    vector_db_emb = None
    if getattr(args, "vector_db_embeddings", None):
        p = Path(str(args.vector_db_embeddings))
        if not p.exists():
            print(f"❌ --vector-db-embeddings not found: {p}")
            return 1
        try:
            obj = np.load(str(p), allow_pickle=True)
            if getattr(obj, "dtype", None) is object and getattr(obj, "shape", ()) == ():
                obj = obj.item()
            if isinstance(obj, dict):
                for k in ("embeddings", "reference_embeddings", "data", "X", "x", "arr"):
                    if k in obj:
                        obj = obj[k]
                        break
            arr = np.asarray(obj)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            if not np.issubdtype(arr.dtype, np.number):
                raise ValueError(f"Embeddings must be numeric, got dtype={arr.dtype}")
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype("float32", copy=False)
            if arr.size == 0 or arr.shape[0] < 1:
                raise ValueError("Embeddings array is empty")
            vector_db_emb = torch.from_numpy(arr).to(dev_pref)
            print(f"✅ Vector DB embeddings loaded: shape={tuple(arr.shape)}")
        except Exception as e:
            print(f"❌ Failed to load --vector-db-embeddings: {e}")
            return 1

    # Create config
    config = EDNNConfig(
        attack_type='rag_poison',
        rag_poison_ratio=args.poison_ratio,
        rag_poison_similarity_threshold=args.similarity_threshold,
        device=dev_pref,
        verbose=bool(getattr(args, "verbose", False)),
        output_dir=args.output_dir
    )
    
    # Initialize EDNN attack
    ednn = EDNNAttack(
        embedding_model=model,
        tokenizer=tokenizer,
        config=config
    )
    
    # Execute RAG poisoning attack
    print("\n⚡ Running retrieval-targeted embedding optimization...")
    print(f"   Target Query: {args.target_query}")
    print(f"   Target top-fraction (rank eval): {args.poison_ratio:.1%}")
    print(f"   Similarity threshold (no DB): {args.similarity_threshold:.2f}")
    print(f"   Vector DB mode: {args.vector_db}")
    if vector_db_emb is None:
        print("   Note: No --vector-db-embeddings provided; reporting query similarity only.")
    
    try:
        result = ednn.rag_poisoning_attack(
            target_query=args.target_query,
            poisoned_document=malicious_content,
            vector_database=vector_db_emb
        )
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        output_file = f"rag_poison_{timestamp}.json"
        ednn.save_result(result, output_file)

        # Also write embeddings for downstream RAG ingestion testing
        try:
            if result.adversarial_embedding is not None:
                np.save(str(out_dir / f"rag_poison_{timestamp}_optimized_embedding.npy"), np.asarray(result.adversarial_embedding))
            np.save(str(out_dir / f"rag_poison_{timestamp}_original_embedding.npy"), np.asarray(result.original_embedding))
        except Exception:
            pass
        
        print(f"\n{'='*80}")
        print("✅ RAG Poisoning Results (analysis output)")
        print(f"{'='*80}")
        print("\n📊 Attack Metrics:")
        print(f"   Success: {'✓' if result.attack_success else '✗'}")
        print(f"   Similarity: {float(result.confidence_score):.4f}")

        meta = result.metadata or {}
        if meta.get("vector_db_size"):
            print("\n🎯 Retrieval Evaluation (local embeddings):")
            print(f"   Vector DB size: {meta.get('vector_db_size')}")
            if meta.get("rank") is not None and meta.get("target_top_k") is not None:
                print(f"   Rank: {meta.get('rank')} (target ≤ {meta.get('target_top_k')})")
            print(f"   Top-fraction: {meta.get('top_fraction')}")
        else:
            print("\n🎯 Retrieval Evaluation:")
            print("   No local vector DB provided; interpret similarity as a proxy only.")
            print(f"   Threshold: {meta.get('similarity_threshold')}")

        print("\n🧾 Notes:")
        print("   - NeurInSpectre does not modify external vector DBs from this command.")
        print("   - To validate impact, insert the produced embedding/doc into your own RAG ingestion pipeline and measure retrieval rank/behavior.")

        print(f"\n💾 Outputs written under: {Path(args.output_dir)}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        import traceback
        print(f"\n❌ Attack failed: {str(e)}")
        if bool(getattr(args, "verbose", False)):
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    
    parser = argparse.ArgumentParser(
        description='EDNN RAG Poisoning Attack - Inject malicious documents into vector databases'
    )
    
    parser.add_argument('--model-path', '-m', required=True, help='Embedding model path')
    parser.add_argument('--vector-db', choices=['weaviate', 'pinecone', 'local'], default='local')
    parser.add_argument('--malicious-doc', '-d', required=True, help='Malicious document path')
    parser.add_argument('--target-query', '-q', required=True, help='Target query')
    parser.add_argument('--poison-ratio', type=float, default=0.1)
    parser.add_argument('--similarity-threshold', type=float, default=0.85)
    parser.add_argument('--output-dir', '-o', default='_cli_runs/ednn_rag_poison')
    parser.add_argument('--verbose', '-v', action='store_true')
    
    args = parser.parse_args()
    sys.exit(run_ednn_rag_poison(args))

