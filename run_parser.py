import json
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from parser import NotionParser
from benchmark import analyze_output, validate_parsing_quality, benchmark_parsing

load_dotenv()

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
PAGE_ID = os.getenv("PAGE_ID")

def main():
    # Initialize parser
    parser = NotionParser(NOTION_TOKEN)
    
    # Parse the document
    result = parser.parse_document(PAGE_ID)
    
    # Save the structured output
    with open("parsed_output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # Convert to LangChain documents for AI workflows
    documents = parser.to_langchain_documents(result)
    
    # Split documents for better chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(documents)
    
    # Save the AI-ready documents
    with open("ai_documents.json", "w", encoding="utf-8") as f:
        # Convert LangChain documents to serializable format
        serializable_docs = []
        for doc in split_docs:
            serializable_docs.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
        json.dump(serializable_docs, f, ensure_ascii=False, indent=2)
    
    # Print metrics
    print("🎉 Parsing complete!")
    print(f"📊 Parsing efficacy: {result['metadata']['parsing_efficacy']}")
    print(f"📦 Total blocks parsed: {result['metadata']['parsed_blocks']}")
    print(f"❌ Failed blocks: {result['metadata']['failed_blocks']}")
    print(f"📎 Embedded files: {result['metadata']['embedded_files']}")
    print("💾 Output saved to parsed_output.json and ai_documents.json")
    
    # Perform analysis
    analysis = analyze_output(result)
    print(f"\n🔍 Content Analysis:")
    print(f"Document Title: {analysis['document_title']}")
    print(f"Total Content Length: {analysis['total_content_length']} characters")
    print(f"Block Type Distribution:")
    for block_type, count in sorted(analysis['block_type_distribution'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {block_type}: {count}")
    
    # Validate parsing quality
    validation = validate_parsing_quality(result)
    print(f"\n✅ Hierarchy preserved: {validation['hierarchy_preserved']}")
    print(f"✅ Metadata preserved: {validation['metadata_preserved']}")
    if validation['errors']:
        print(f"❌ Errors: {validation['errors']}")
    
    # Benchmark performance
    benchmark = benchmark_parsing(parser, PAGE_ID)
    print(f"\n⏱️  Parsing Time: {benchmark['parsing_time_seconds']}s")
    print(f"📦 Blocks per Second: {benchmark['blocks_per_second']}")
    print(f"💾 Memory Usage: {benchmark['memory_usage_mb']} MB")
    
    return parser, result

if __name__ == "__main__":
    parser, result = main()