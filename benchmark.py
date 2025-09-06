import json
import time
import psutil
from typing import Dict, Any, List
from parser import NotionParser

def analyze_output(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the parsed output from NotionParser for document metrics and insights.

    Args:
        parsed_data (Dict[str, Any]): The parsed Notion document data.

    Returns:
        Dict[str, Any]: Dictionary containing document title, type, creation and edit times, block type distribution, and total content length.
    """
    
    analysis = {
        "document_title": parsed_data.get('title', 'N/A'),
        "document_type": parsed_data.get('type', 'N/A'),
        "created_time": parsed_data.get('created_time', 'N/A'),
        "last_edited_time": parsed_data.get('last_edited_time', 'N/A'),
        "block_type_distribution": {},
        "total_content_length": 0
    }
    
    # Count block types
    def count_blocks(blocks):
        for block in blocks:
            block_type = block.get('type', 'unknown')
            analysis["block_type_distribution"][block_type] = analysis["block_type_distribution"].get(block_type, 0) + 1
            
            # Count content length
            if block.get('content'):
                analysis["total_content_length"] += len(block['content'])
                
            if 'children' in block:
                count_blocks(block['children'])
    
    if 'content' in parsed_data and parsed_data['content']:
        count_blocks(parsed_data['content'])
    
    return analysis

def validate_parsing_quality(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the quality of parsing by checking hierarchy and metadata preservation against the original Notion structure.

    Args:
        parsed_data (Dict[str, Any]): The parsed Notion document data.

    Returns:
        Dict[str, Any]: Dictionary indicating if hierarchy and metadata are preserved, and listing any errors found.
    """
    
    validation_results = {
        "hierarchy_preserved": True,
        "metadata_preserved": True,
        "errors": []
    }
    
    # Check hierarchy preservation
    def check_hierarchy(blocks, depth=0):
        for block in blocks:
            if block.get('depth', 0) != depth:
                validation_results["hierarchy_preserved"] = False
                validation_results["errors"].append(f"Depth mismatch in {block.get('id', 'unknown')}")
            
            if 'children' in block:
                check_hierarchy(block['children'], depth + 1)
    
    if parsed_data.get('content'):
        check_hierarchy(parsed_data['content'])
    
    # Check metadata preservation
    if not parsed_data.get('title'):
        validation_results["metadata_preserved"] = False
        validation_results["errors"].append("Title not preserved")
    
    if not parsed_data.get('created_time'):
        validation_results["metadata_preserved"] = False
        validation_results["errors"].append("Created time not preserved")
    
    if not parsed_data.get('last_edited_time'):
        validation_results["metadata_preserved"] = False
        validation_results["errors"].append("Last edited time not preserved")
    
    return validation_results

def benchmark_parsing(parser: NotionParser, page_id: str) -> Dict[str, Any]:
    """
    Benchmark the performance of the NotionParser on a single Notion page or database.

    Args:
        parser (NotionParser): The NotionParser instance.
        page_id (str): The Notion page or database ID to parse.

    Returns:
        Dict[str, Any]: Dictionary containing parsing time, blocks per second, memory usage, parsed blocks, failed blocks, embedded files, and parsing efficacy.
    """
    
    print("\n⏱️  Running performance benchmark...")
    start_time = time.time()
    
    # Parse the document for benchmarking
    result = parser.parse_document(page_id)
    
    parsing_time = time.time() - start_time
    
    # Memory usage (approximate)
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    benchmark_results = {
        "parsing_time_seconds": round(parsing_time, 2),
        "blocks_per_second": round(result['metadata']['parsed_blocks'] / parsing_time, 2) if parsing_time > 0 else 0,
        "memory_usage_mb": round(memory_mb, 2),
        "parsed_blocks": result['metadata']['parsed_blocks'],
        "failed_blocks": result['metadata']['failed_blocks'],
        "embedded_files": result['metadata']['embedded_files'],
        "parsing_efficacy": result['metadata']['parsing_efficacy']
    }
    
    return benchmark_results

def benchmark_multiple_docs(parser: NotionParser, page_ids: List[str]) -> Dict[str, Any]:
    """
    Benchmark the NotionParser on multiple Notion documents and aggregate the results.

    Args:
        parser (NotionParser): The NotionParser instance.
        page_ids (List[str]): List of Notion page or database IDs to parse.

    Returns:
        Dict[str, Any]: Dictionary containing individual benchmark results and summary statistics (averages, success/failure counts).
    """
    
    results = []
    
    for page_id in page_ids:
        try:
            benchmark_result = benchmark_parsing(parser, page_id)
            results.append({
                "page_id": page_id,
                **benchmark_result
            })
        except Exception as e:
            results.append({
                "page_id": page_id,
                "error": str(e)
            })
    
    # Calculate averages
    successful_results = [r for r in results if "error" not in r]
    
    if successful_results:
        avg_parsing_time = sum(r["parsing_time_seconds"] for r in successful_results) / len(successful_results)
        avg_blocks_per_second = sum(r["blocks_per_second"] for r in successful_results) / len(successful_results)
        avg_memory_usage = sum(r["memory_usage_mb"] for r in successful_results) / len(successful_results)
    else:
        avg_parsing_time = avg_blocks_per_second = avg_memory_usage = 0
    
    return {
        "individual_results": results,
        "summary": {
            "total_documents": len(page_ids),
            "successful_parses": len(successful_results),
            "failed_parses": len(results) - len(successful_results),
            "average_parsing_time_seconds": round(avg_parsing_time, 2),
            "average_blocks_per_second": round(avg_blocks_per_second, 2),
            "average_memory_usage_mb": round(avg_memory_usage, 2)
        }
    }