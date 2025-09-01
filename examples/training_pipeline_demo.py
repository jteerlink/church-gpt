"""
Training Pipeline Demo
Demonstrates end-to-end processing from raw documents to training data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.training_pipeline import (
    TrainingDataPipeline, create_training_pipeline, 
    process_conference_content_to_training_data
)


def main():
    """Run training pipeline demonstration."""
    print("🚀 Training Pipeline Demo")
    print("=" * 50)
    
    # Sample conference documents
    sample_documents = [
        {
            'content': '''Faith is not only a feeling; it is a decision. Faith is choosing to do what the Lord has asked us to do. When we choose to act with faith—especially when we do not feel very faithful—the Lord sees our effort.

The prophet Alma taught his son Helaman about the power of faith when he said: "And now, my son, I would that ye should understand that these things are not without a shadow."

Brothers and sisters, faith is not passive. Faith is not hoping that things will work out. Faith is a principle of action and power.''',
            'author': 'Dieter F. Uchtdorf',
            'title': 'Acting on Our Faith',
            'date': '2020-04-01',
            'source': 'general-conference/2020-04/acting-on-our-faith.txt'
        },
        {
            'content': '''I know that Jesus Christ lives. I testify that He is our Savior and Redeemer. My testimony has been strengthened through personal revelation and the witness of the Holy Ghost.

When I was young, I learned about the power of prayer through a personal experience that changed my life forever. This experience taught me that Heavenly Father knows each of us individually.''',
            'author': 'Russell M. Nelson',
            'title': 'My Testimony of the Savior',
            'date': '2020-10-01',
            'source': 'general-conference/2020-10/my-testimony-of-the-savior.txt'
        }
    ]
    
    print(f"📚 Input Documents: {len(sample_documents)}")
    for i, doc in enumerate(sample_documents, 1):
        print(f"   {i}. {doc['author']}: '{doc['title']}'")
    print()
    
    # Create training pipeline
    print("🔧 Creating Training Pipeline...")
    pipeline = create_training_pipeline(
        tokenizer_name="bert-base-uncased",  # Use public tokenizer for demo
        max_tokens=200,
        chunking_strategy="paragraph_aware"
    )
    
    # Process documents
    print("⚙️ Processing Documents...")
    training_examples = pipeline.process_documents_to_training_data(sample_documents)
    
    print(f"✅ Generated {len(training_examples)} training examples")
    print()
    
    # Show sample training example
    if training_examples:
        example = training_examples[0]
        print("📝 Sample Training Example:")
        print("-" * 40)
        print("🎯 PROMPT:")
        print(example['prompt'][:300] + "..." if len(example['prompt']) > 300 else example['prompt'])
        print()
        print("📖 RESPONSE:")
        print(example['response'][:200] + "..." if len(example['response']) > 200 else example['response'])
        print()
        print("📊 METADATA:")
        metadata = example['metadata']
        print(f"   Author: {metadata['author']}")
        print(f"   Content Type: {metadata['content_type']}")
        print(f"   Template Used: {metadata['template_used']}")
        print(f"   Token Count: {metadata['token_count']}")
        print()
    
    # Generate comprehensive report
    print("📊 Generating Pipeline Report...")
    report = pipeline.generate_pipeline_report(sample_documents, training_examples)
    
    print("📈 Pipeline Summary:")
    print(f"   Input Documents: {report['pipeline_summary']['input_documents']}")
    print(f"   Training Examples: {report['pipeline_summary']['training_examples_generated']}")
    print(f"   Expansion Ratio: {report['pipeline_summary']['expansion_ratio']:.1f}x")
    print(f"   Avg Prompt Length: {report['pipeline_summary']['avg_prompt_length']:.0f} chars")
    print(f"   Avg Response Length: {report['pipeline_summary']['avg_response_length']:.0f} chars")
    print()
    
    print("🎯 Quality Metrics:")
    quality = report['quality_indicators']
    print(f"   Boundary Preservation: {quality['chunking_boundary_preservation']:.2f}")
    print(f"   Token Utilization: {quality['token_utilization']:.2f}")
    print(f"   Template Diversity: {quality['prompt_template_diversity']:.2f}")
    print()
    
    print("📋 Content Analysis:")
    prompt_analysis = report['prompt_analysis']
    print(f"   Content Types Found: {list(prompt_analysis['content_type_distribution'].keys())}")
    print(f"   Templates Used: {list(prompt_analysis['template_usage'].keys())}")
    print(f"   Template Diversity Score: {prompt_analysis['template_diversity']}")
    print()
    
    # Save training data (optional)
    output_file = "training_data_demo.jsonl"
    pipeline.save_training_data(training_examples, output_file)
    print(f"💾 Training data saved to: {output_file}")
    
    print("✅ Training Pipeline Demo Complete!")


if __name__ == "__main__":
    main()