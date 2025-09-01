"""
Demo script for Smart Chunking System
Demonstrates chunking capabilities with sample General Conference content.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.chunking import SmartChunker, process_documents_batch


def main():
    """Run chunking demonstration."""
    print("🔧 Smart Chunking System Demo")
    print("=" * 50)
    
    # Sample conference content
    sample_document = {
        'content': '''Faith is not only a feeling; it is a decision. Faith is choosing to do what the Lord has asked us to do. When we choose to act with faith—especially when we do not feel very faithful—the Lord sees our effort.

The prophet Alma taught his son Helaman about the power of faith when he said: "And now, my son, I would that ye should understand that these things are not without a shadow; for as our fathers were slothful to give heed unto this compass."

Brothers and sisters, faith is not passive. Faith is not hoping that things will work out. Faith is not wishful thinking. Faith is a principle of action and power.''',
        'author': 'Dieter F. Uchtdorf',
        'title': 'Acting on Our Faith',
        'date': '2020-04-01',
        'source': 'demo_content.txt'
    }
    
    print(f"📄 Original Document: {len(sample_document['content'])} characters")
    print(f"👤 Author: {sample_document['author']}")
    print(f"📅 Date: {sample_document['date']}")
    print()
    
    # Test different chunking strategies
    strategies = ["sentences", "paragraph_aware", "overlap"]
    
    for strategy in strategies:
        print(f"🧩 Testing {strategy.upper()} strategy:")
        print("-" * 40)
        
        try:
            # Use a public tokenizer for demo
            chunker = SmartChunker("bert-base-uncased", max_tokens=200)
            chunk_docs = chunker.process_document(sample_document, strategy=strategy)
            
            print(f"📊 Generated {len(chunk_docs)} chunks")
            
            # Show first chunk as example
            if chunk_docs:
                first_chunk = chunk_docs[0]
                print(f"📝 First chunk ({first_chunk['token_count']} tokens):")
                print(f"   {first_chunk['content'][:100]}...")
            
            # Calculate quality metrics
            chunks_content = [chunk['content'] for chunk in chunk_docs]
            metrics = chunker.calculate_chunk_quality_score(chunks_content)
            
            print(f"📈 Quality Metrics:")
            print(f"   Token Utilization: {metrics.get('token_utilization', 0):.2f}")
            print(f"   Boundary Preservation: {metrics.get('boundary_preservation', 0):.2f}")
            print(f"   Size Consistency: {metrics.get('size_consistency', 0):.2f}")
            print()
            
        except Exception as e:
            print(f"❌ Error with {strategy}: {e}")
            print()
    
    # Batch processing demo
    print("📦 Batch Processing Demo:")
    print("-" * 40)
    
    documents = [sample_document, {
        'content': 'Second document for batch testing. This contains different content.',
        'author': 'Test Author',
        'title': 'Batch Test Document',
        'date': '2024-01-01',
        'source': 'batch_test.txt'
    }]
    
    try:
        chunks, report = process_documents_batch(
            documents,
            tokenizer_name="bert-base-uncased",
            max_tokens=150,
            strategy="paragraph_aware"
        )
        
        print(f"📊 Batch Results:")
        print(f"   Original Documents: {report['summary']['original_docs']}")
        print(f"   Total Chunks: {report['summary']['total_chunks']}")
        print(f"   Chunks per Document: {report['summary']['chunks_per_doc']:.1f}")
        print(f"   Token Preservation: {report['summary']['token_preservation']:.2f}")
        print(f"   Average Chunk Size: {report['quality_metrics']['avg_chunk_size']:.1f} tokens")
        print()
        
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
    
    print("✅ Demo completed!")


if __name__ == "__main__":
    main()