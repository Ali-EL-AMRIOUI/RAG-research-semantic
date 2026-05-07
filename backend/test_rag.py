#!/usr/bin/env python3

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.rag_service import IndustrialRAGService
from app.core.config import get_config_dict


def test_initialization():
    print("\n" + "=" * 60)
    print("TEST 1: RAG Service Initialization")
    print("=" * 60)
    
    try:
        rag = IndustrialRAGService()
        print("RAG service initialized successfully")
        return rag
    except Exception as e:
        print(f"Initialization error: {e}")
        return None


def test_config():
    print("\n" + "=" * 60)
    print("TEST 2: Configuration")
    print("=" * 60)
    
    config = get_config_dict()
    for key, value in config.items():
        print(f"   {key}: {value}")
    print("Configuration loaded")


def test_simple_question(rag: IndustrialRAGService, question: str = None):
    print("\n" + "=" * 60)
    print("TEST 3: Simple Question")
    print("=" * 60)
    
    if question is None:
        question = "What is machine learning?"
    
    print(f"Question: {question}")
    print("-" * 40)
    
    try:
        start_time = time.time()
        response = rag.run(question)
        elapsed_ms = (time.time() - start_time) * 1000
        
        print(f"\nAnswer:\n{response['answer']}")
        print("-" * 40)
        print(f"Sources: {len(response.get('source_documents', []))} chunks")
        print(f"Time: {elapsed_ms:.2f} ms")
        
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_search_only(rag: IndustrialRAGService, query: str = None):
    print("\n" + "=" * 60)
    print("TEST 4: Search without generation")
    print("=" * 60)
    
    if query is None:
        query = "artificial intelligence"
    
    print(f"Search: {query}")
    print("-" * 40)
    
    try:
        results = rag.search(query, limit=5)
        print(f"{len(results)} results found")
        
        for i, r in enumerate(results):
            print(f"\n   [{i+1}] Score: {r['score']:.4f}")
            print(f"       Filename: {r['filename']}")
            print(f"       Text: {r['text'][:150]}...")
        
        return results
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_documents_list(rag: IndustrialRAGService):
    print("\n" + "=" * 60)
    print("TEST 5: Documents List")
    print("=" * 60)
    
    try:
        documents = rag.get_all_documents()
        
        if not documents:
            print("No documents indexed")
            print("Use /upload API to index documents")
            return []
        
        print(f"{len(documents)} document(s) indexed:")
        for doc in documents:
            if isinstance(doc, dict):
                name = doc.get('name', doc.get('filename', str(doc)))
                chunks = doc.get('chunks', 0)
                print(f"   {name} ({chunks} chunks)")
            else:
                print(f"   {doc}")
        
        return documents
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_stats(rag: IndustrialRAGService):
    print("\n" + "=" * 60)
    print("TEST 6: Statistics")
    print("=" * 60)
    
    try:
        stats = rag.get_stats()
        print(f"   Collection: {stats.get('collection_name', 'unknown')}")
        print(f"   Points: {stats.get('total_points', 0)}")
        print(f"   Vectors: {stats.get('total_vectors', 0)}")
        print(f"   Documents: {stats.get('documents_count', 0)}")
        return stats
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_custom_question(rag: IndustrialRAGService):
    print("\n" + "=" * 60)
    print("TEST 7: Interactive Mode")
    print("=" * 60)
    print("Ask questions (type 'quit' to exit)")
    
    while True:
        print("\n" + "-" * 40)
        question = input("Question: ").strip()
        
        if question.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        try:
            start_time = time.time()
            response = rag.run(question)
            elapsed_ms = (time.time() - start_time) * 1000
            
            print(f"\nAnswer:\n{response['answer']}")
            print(f"\nTime: {elapsed_ms:.2f} ms")
            
        except Exception as e:
            print(f"Error: {e}")


def run_all_tests():
    print("\n" + "=" * 60)
    print("STARTING RAG TESTS")
    print("=" * 60)
    
    test_config()
    
    rag = test_initialization()
    if not rag:
        print("\nCannot continue tests")
        return
    
    test_documents_list(rag)
    test_stats(rag)
    
    test_search_only(rag)
    
    test_simple_question(rag)
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    
    print("\nSwitch to interactive mode? (y/n)")
    if input().lower() in ('y', 'yes'):
        test_custom_question(rag)


def quick_test():
    print("\nQuick RAG test")
    
    rag = test_initialization()
    if not rag:
        return
    
    question = "What is the main objective of this project?"
    print(f"Question: {question}")
    
    response = rag.run(question)
    print(f"\nAnswer:\n{response['answer']}")
    print(f"\nSources: {len(response.get('source_documents', []))} chunks")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Service Test")
    parser.add_argument("--quick", action="store_true", help="Quick test")
    parser.add_argument("--question", "-q", type=str, help="Custom question")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    elif args.question:
        rag = test_initialization()
        if rag:
            test_simple_question(rag, args.question)
    elif args.interactive:
        rag = test_initialization()
        if rag:
            test_custom_question(rag)
    else:
        run_all_tests()