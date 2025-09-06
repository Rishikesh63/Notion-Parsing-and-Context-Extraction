import json
import re
import requests
from typing import Dict, List, Any, Optional, Tuple
from notion_client import Client
from notion_client.errors import APIResponseError
from langchain.schema import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import io
import csv
import json
from notion_client import Client
from notion_client.errors import APIResponseError
import PyPDF2
from PIL import Image
import pytesseract
import logging
from typing import Dict, List, Any, Optional

class NotionParser:
    """
    NotionParser parses Notion documents and databases, supporting multiple block types and extensible via the Strategy pattern.
    """
    def __init__(self, token: str, ocr_enabled: bool = True, max_retries: int = 3):
        self.client = Client(auth=token)
        self.token = token
        self.ocr_enabled = ocr_enabled
        self.max_retries = max_retries
        self.parsed_blocks_count = 0
        self.failed_blocks_count = 0
        self.embedded_files_count = 0
        # Strategy registry for block parsing
        self.block_parsers = {
            "paragraph": self._parse_paragraph,
            "heading_1": self._parse_heading,
            "heading_2": self._parse_heading,
            "heading_3": self._parse_heading,
            "bulleted_list_item": self._parse_list_item,
            "numbered_list_item": self._parse_list_item,
            "to_do": self._parse_to_do,
            "toggle": self._parse_toggle,
            "code": self._parse_code,
            "quote": self._parse_quote,
            "callout": self._parse_callout,
            "table": self._parse_table,
            "table_row": self._parse_table_row,
            "image": self._parse_image,
            "file": self._parse_file,
            "pdf": self._parse_pdf,
            "video": self._parse_video,
            "bookmark": self._parse_bookmark,
            "embed": self._parse_embed,
            "equation": self._parse_equation,
            "divider": self._parse_divider,
            "table_of_contents": self._parse_table_of_contents,
            "child_page": self._parse_child_page,
            "child_database": self._parse_child_database,
        }
        self.ocr_enabled = ocr_enabled
        self.max_retries = max_retries
        
    def parse_document(self, page_or_db_id: str) -> Dict[str, Any]:
        """
        Parse a Notion document with comprehensive structure preservation
        """
        # Initialize metrics
        self.parsed_blocks_count = 0
        self.failed_blocks_count = 0
        self.embedded_files_count = 0
        
        # Get the root page/database
        root_object = self._get_root_object(page_or_db_id)
        
        # Parse all blocks recursively
        parsed_content = self._parse_blocks_recursively(page_or_db_id)
        
        # Calculate metrics
        total_blocks = self.parsed_blocks_count + self.failed_blocks_count
        parsing_efficacy = (self.parsed_blocks_count / total_blocks * 100) if total_blocks > 0 else 0
        
        result = {
            "id": page_or_db_id,
            "type": root_object.get("object", "unknown"),
            "title": self._extract_title(root_object),
            "url": root_object.get("url", ""),
            "created_time": root_object.get("created_time", ""),
            "last_edited_time": root_object.get("last_edited_time", ""),
            "content": parsed_content,
            "metadata": {
                "parsed_blocks": self.parsed_blocks_count,
                "failed_blocks": self.failed_blocks_count,
                "embedded_files": self.embedded_files_count,
                "parsing_efficacy": f"{parsing_efficacy:.2f}%",
                "total_blocks": total_blocks
            }
        }
        
        return result
    
    def _get_root_object(self, page_or_db_id: str) -> Dict[str, Any]:
        """Retrieve the root page or database object with retry logic"""
        for attempt in range(self.max_retries):
            try:
                # Try as page
                return self.client.pages.retrieve(page_or_db_id)
            except APIResponseError as e:
                if "rate limited" in str(e).lower() and attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                try:
                    # Try as database
                    return self.client.databases.retrieve(page_or_db_id)
                except APIResponseError as e2:
                    if "rate limited" in str(e2).lower() and attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    try:
                        # Try as block
                        return self.client.blocks.retrieve(page_or_db_id)
                    except APIResponseError as e3:
                        if attempt == self.max_retries - 1:
                            raise Exception(f"Failed to retrieve object {page_or_db_id} after {self.max_retries} attempts: {e3}")
                        time.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception(f"Failed to retrieve object {page_or_db_id}")
    
    def _extract_title(self, notion_object: Dict[str, Any]) -> str:
        """Extract title from a Notion object"""
        title = ""
        if notion_object.get("object") == "page":
            # Page title is in the properties
            properties = notion_object.get("properties", {})
            for prop_name, prop_value in properties.items():
                if prop_value.get("type") == "title" and prop_value.get("title"):
                    title = " ".join([t.get("plain_text", "") for t in prop_value.get("title", [])])
                    break
        elif notion_object.get("object") == "database":
            # Database title
            title_array = notion_object.get("title", [])
            title = " ".join([t.get("plain_text", "") for t in title_array])
        
        return title
    
    def _parse_blocks_recursively(self, block_id: str, depth: int = 0) -> List[Dict[str, Any]]:
        """Recursively parse all blocks with hierarchy preservation"""
        for attempt in range(self.max_retries):
            try:
                blocks_response = self.client.blocks.children.list(block_id)
                blocks = blocks_response.get("results", [])
                parsed_blocks = []
                
                for block in blocks:
                    parsed_block = self._parse_block(block, depth)
                    if parsed_block:
                        parsed_blocks.append(parsed_block)
                        
                        # Recursively parse child blocks if they exist
                        if block.get("has_children", False):
                            children = self._parse_blocks_recursively(block["id"], depth + 1)
                            if children:
                                parsed_block["children"] = children
                
                return parsed_blocks
            except APIResponseError as e:
                if "rate limited" in str(e).lower() and attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                logger.error(f"Error fetching blocks for {block_id}: {e}")
                self.failed_blocks_count += 1
                return []
        
        logger.error(f"Failed to fetch blocks for {block_id} after {self.max_retries} attempts")
        self.failed_blocks_count += 1
        return []
    
    def _parse_block(self, block: Dict[str, Any], depth: int = 0) -> Optional[Dict[str, Any]]:
        """Parse an individual block using the strategy pattern"""
        block_type = block.get("type")
        block_id = block.get("id")
        parser_func = self.block_parsers.get(block_type, self._parse_unsupported)
        parsed_block = parser_func(block, depth)
        self.parsed_blocks_count += 1
        return parsed_block
    # Strategy implementations for each block type
    def _parse_paragraph(self, block, depth):
        return {
            "id": block.get("id"),
            "type": "paragraph",
            "depth": depth,
            "content": self._extract_rich_text(block["paragraph"].get("rich_text", [])),
            "metadata": {}
        }

    def _parse_heading(self, block, depth):
        block_type = block.get("type")
        return {
            "id": block.get("id"),
            "type": block_type,
            "depth": depth,
            "content": self._extract_rich_text(block[block_type].get("rich_text", [])),
            "metadata": {"level": int(block_type.split("_")[1])}
        }

    def _parse_list_item(self, block, depth):
        block_type = block.get("type")
        return {
            "id": block.get("id"),
            "type": block_type,
            "depth": depth,
            "content": self._extract_rich_text(block[block_type].get("rich_text", [])),
            "metadata": {}
        }

    def _parse_to_do(self, block, depth):
        return {
            "id": block.get("id"),
            "type": "to_do",
            "depth": depth,
            "content": self._extract_rich_text(block["to_do"].get("rich_text", [])),
            "metadata": {"checked": block["to_do"].get("checked", False)}
        }

    def _parse_toggle(self, block, depth):
        return {
            "id": block.get("id"),
            "type": "toggle",
            "depth": depth,
            "content": self._extract_rich_text(block["toggle"].get("rich_text", [])),
            "metadata": {}
        }

    def _parse_code(self, block, depth):
        return {
            "id": block.get("id"),
            "type": "code",
            "depth": depth,
            "content": self._extract_rich_text(block["code"].get("rich_text", [])),
            "metadata": {"language": block["code"].get("language", "")}
        }

    def _parse_quote(self, block, depth):
        return {
            "id": block.get("id"),
            "type": "quote",
            "depth": depth,
            "content": self._extract_rich_text(block["quote"].get("rich_text", [])),
            "metadata": {}
        }

    def _parse_callout(self, block, depth):
        meta = {}
        if "icon" in block["callout"]:
            meta["icon"] = block["callout"]["icon"]
        return {
            "id": block.get("id"),
            "type": "callout",
            "depth": depth,
            "content": self._extract_rich_text(block["callout"].get("rich_text", [])),
            "metadata": meta
        }

    def _parse_table(self, block, depth):
        # ...existing code...
        return {"id": block.get("id"), "type": "table", "depth": depth, "content": None, "metadata": {}}

    def _parse_table_row(self, block, depth):
        # ...existing code...
        return {"id": block.get("id"), "type": "table_row", "depth": depth, "content": None, "metadata": {}}

    def _parse_image(self, block, depth):
        # ...existing code...
        return {"id": block.get("id"), "type": "image", "depth": depth, "content": None, "metadata": {}}

    def _parse_file(self, block, depth):
        # ...existing code...
        return {"id": block.get("id"), "type": "file", "depth": depth, "content": None, "metadata": {}}

    def _parse_pdf(self, block, depth):
        # ...existing code...
        return {"id": block.get("id"), "type": "pdf", "depth": depth, "content": None, "metadata": {}}

    def _parse_video(self, block, depth):
        # ...existing code...
        return {"id": block.get("id"), "type": "video", "depth": depth, "content": None, "metadata": {}}

    def _parse_bookmark(self, block, depth):
        # ...existing code...
        return {"id": block.get("id"), "type": "bookmark", "depth": depth, "content": None, "metadata": {}}

    def _parse_embed(self, block, depth):
        # ...existing code...
        return {"id": block.get("id"), "type": "embed", "depth": depth, "content": None, "metadata": {}}

    def _parse_equation(self, block, depth):
        return {
            "id": block.get("id"),
            "type": "equation",
            "depth": depth,
            "content": block["equation"].get("expression", ""),
            "metadata": {}
        }

    def _parse_divider(self, block, depth):
        return {
            "id": block.get("id"),
            "type": "divider",
            "depth": depth,
            "content": "---",
            "metadata": {}
        }

    def _parse_table_of_contents(self, block, depth):
        return {
            "id": block.get("id"),
            "type": "table_of_contents",
            "depth": depth,
            "content": "Table of Contents",
            "metadata": {}
        }

    def _parse_child_page(self, block, depth):
        parsed_block = {
            "id": block.get("id"),
            "type": "child_page",
            "depth": depth,
            "content": block["child_page"].get("title", ""),
            "metadata": {}
        }
        try:
            child_blocks = self._parse_blocks_recursively(block.get("id"), depth + 1)
            if child_blocks:
                parsed_block["children"] = child_blocks
        except Exception as e:
            logger.error(f"Failed to parse child page {block.get('id')}: {e}")
        return parsed_block

    def _parse_child_database(self, block, depth):
        parsed_block = {
            "id": block.get("id"),
            "type": "child_database",
            "depth": depth,
            "content": block["child_database"].get("title", ""),
            "metadata": {}
        }
        try:
            db_content = self._parse_database(block.get("id"))
            if db_content:
                parsed_block["children"] = db_content
        except Exception as e:
            logger.error(f"Failed to parse database {block.get('id')}: {e}")
        return parsed_block

    def _parse_unsupported(self, block, depth):
        block_type = block.get("type")
        return {
            "id": block.get("id"),
            "type": block_type,
            "depth": depth,
            "content": f"[Unsupported block type: {block_type}]",
            "metadata": {}
        }

        # Strategy implementations for each block type
    def _parse_paragraph(self, block, depth):
        return {
            "id": block.get("id"),
            "type": "paragraph",
            "depth": depth,
            "content": self._extract_rich_text(block["paragraph"].get("rich_text", [])),
            "metadata": {}
        }

    def _parse_heading(self, block, depth):
        block_type = block.get("type")
        return {
            "id": block.get("id"),
            "type": block_type,
            "depth": depth,
            "content": self._extract_rich_text(block[block_type].get("rich_text", [])),
            "metadata": {"level": int(block_type.split("_")[1])}
        }

    def _parse_list_item(self, block, depth):
        block_type = block.get("type")
        return {
            "id": block.get("id"),
            "type": block_type,
            "depth": depth,
            "content": self._extract_rich_text(block[block_type].get("rich_text", [])),
            "metadata": {}
        }

    def _parse_to_do(self, block, depth):
        return {
            "id": block.get("id"),
            "type": "to_do",
            "depth": depth,
            "content": self._extract_rich_text(block["to_do"].get("rich_text", [])),
            "metadata": {"checked": block["to_do"].get("checked", False)}
        }

    def _parse_toggle(self, block, depth):
        return {
            "id": block.get("id"),
            "type": "toggle",
            "depth": depth,
            "content": self._extract_rich_text(block["toggle"].get("rich_text", [])),
            "metadata": {}
        }

    def _parse_code(self, block, depth):
        return {
            "id": block.get("id"),
            "type": "code",
            "depth": depth,
            "content": self._extract_rich_text(block["code"].get("rich_text", [])),
            "metadata": {"language": block["code"].get("language", "")}
        }

    def _parse_quote(self, block, depth):
        return {
            "id": block.get("id"),
            "type": "quote",
            "depth": depth,
            "content": self._extract_rich_text(block["quote"].get("rich_text", [])),
            "metadata": {}
        }

    def _parse_callout(self, block, depth):
        meta = {}
        if "icon" in block["callout"]:
            meta["icon"] = block["callout"]["icon"]
        return {
            "id": block.get("id"),
            "type": "callout",
            "depth": depth,
            "content": self._extract_rich_text(block["callout"].get("rich_text", [])),
            "metadata": meta
        }

    def _parse_table(self, block, depth):
        # TODO: Implement table parsing logic
        return {"id": block.get("id"), "type": "table", "depth": depth, "content": None, "metadata": {}}

    def _parse_table_row(self, block, depth):
        # TODO: Implement table row parsing logic
        return {"id": block.get("id"), "type": "table_row", "depth": depth, "content": None, "metadata": {}}

    def _parse_image(self, block, depth):
        # TODO: Implement image parsing logic
        return {"id": block.get("id"), "type": "image", "depth": depth, "content": None, "metadata": {}}

    def _parse_file(self, block, depth):
        # TODO: Implement file parsing logic
        return {"id": block.get("id"), "type": "file", "depth": depth, "content": None, "metadata": {}}

    def _parse_pdf(self, block, depth):
        # TODO: Implement PDF parsing logic
        return {"id": block.get("id"), "type": "pdf", "depth": depth, "content": None, "metadata": {}}

    def _parse_video(self, block, depth):
        # TODO: Implement video parsing logic
        return {"id": block.get("id"), "type": "video", "depth": depth, "content": None, "metadata": {}}

    def _parse_bookmark(self, block, depth):
        # TODO: Implement bookmark parsing logic
        return {"id": block.get("id"), "type": "bookmark", "depth": depth, "content": None, "metadata": {}}

    def _parse_embed(self, block, depth):
        # TODO: Implement embed parsing logic
        return {"id": block.get("id"), "type": "embed", "depth": depth, "content": None, "metadata": {}}

    def _parse_equation(self, block, depth):
        return {
            "id": block.get("id"),
            "type": "equation",
            "depth": depth,
            "content": block["equation"].get("expression", ""),
            "metadata": {}
        }

    def _parse_divider(self, block, depth):
        return {
            "id": block.get("id"),
            "type": "divider",
            "depth": depth,
            "content": "---",
            "metadata": {}
        }

    def _parse_table_of_contents(self, block, depth):
        return {
            "id": block.get("id"),
            "type": "table_of_contents",
            "depth": depth,
            "content": "Table of Contents",
            "metadata": {}
        }

    def _parse_child_page(self, block, depth):
        parsed_block = {
            "id": block.get("id"),
            "type": "child_page",
            "depth": depth,
            "content": block["child_page"].get("title", ""),
            "metadata": {}
        }
        try:
            child_blocks = self._parse_blocks_recursively(block.get("id"), depth + 1)
            if child_blocks:
                parsed_block["children"] = child_blocks
        except Exception as e:
            logger.error(f"Failed to parse child page {block.get('id')}: {e}")
        return parsed_block

    def _parse_child_database(self, block, depth):
        parsed_block = {
            "id": block.get("id"),
            "type": "child_database",
            "depth": depth,
            "content": block["child_database"].get("title", ""),
            "metadata": {}
        }
        try:
            db_content = self._parse_database(block.get("id"))
            if db_content:
                parsed_block["children"] = db_content
        except Exception as e:
            logger.error(f"Failed to parse database {block.get('id')}: {e}")
        return parsed_block

    def _parse_unsupported(self, block, depth):
        block_type = block.get("type")
        return {
            "id": block.get("id"),
            "type": block_type,
            "depth": depth,
            "content": f"[Unsupported block type: {block_type}]",
            "metadata": {}
        }
        
    
    def _parse_database(self, database_id: str) -> List[Dict[str, Any]]:
        """Parse a Notion database"""
        try:
            # Query the database
            response = self.client.databases.query(database_id)
            pages = response.get("results", [])
            
            parsed_pages = []
            for page in pages:
                parsed_page = {
                    "id": page["id"],
                    "type": "database_page",
                    "depth": 0,
                    "content": self._extract_title(page),
                    "metadata": {
                        "created_time": page.get("created_time", ""),
                        "last_edited_time": page.get("last_edited_time", "")
                    }
                }
                
                # Parse the page content
                page_content = self._parse_blocks_recursively(page["id"], 1)
                if page_content:
                    parsed_page["children"] = page_content
                
                parsed_pages.append(parsed_page)
            
            return parsed_pages
        except Exception as e:
            logger.error(f"Error parsing database {database_id}: {e}")
            return []
    
    def _extract_rich_text(self, rich_text_array: List[Dict[str, Any]]) -> str:
        """Extract plain text from rich text array"""
        text_parts = []
        for text_item in rich_text_array:
            text_content = text_item.get("plain_text", "")
            text_type = text_item.get("type", "text")
            
            # Handle annotations (bold, italic, etc.)
            annotations = text_item.get("annotations", {})
            if annotations.get("bold"):
                text_content = f"**{text_content}**"
            if annotations.get("italic"):
                text_content = f"*{text_content}*"
            if annotations.get("strikethrough"):
                text_content = f"~~{text_content}~~"
            if annotations.get("underline"):
                text_content = f"__{text_content}__"
            if annotations.get("code"):
                text_content = f"`{text_content}`"
            
            # Handle links
            if text_item.get("href"):
                text_content = f"[{text_content}]({text_item['href']})"
            
            # Handle mentions
            if text_type == "mention":
                mention = text_item.get("mention", {})
                mention_type = mention.get("type")
                if mention_type == "user":
                    user = mention.get("user", {})
                    name = user.get("name", "Unknown User")
                    text_content = f"@{name}"
                elif mention_type == "page":
                    page_id = mention.get("page", {}).get("id", "")
                    text_content = f"[Page](https://www.notion.so/{page_id.replace('-', '')})"
                elif mention_type == "database":
                    database_id = mention.get("database", {}).get("id", "")
                    text_content = f"[Database](https://www.notion.so/{database_id.replace('-', '')})"
                elif mention_type == "date":
                    date = mention.get("date", {})
                    start = date.get("start", "")
                    end = date.get("end", "")
                    if end:
                        text_content = f"{start} to {end}"
                    else:
                        text_content = start
            
            text_parts.append(text_content)
        
        return "".join(text_parts)
    
    def _parse_table(self, block: Dict[str, Any], depth: int) -> Dict[str, Any]:
        """Parse a table block"""
        table_data = block["table"]
        parsed_block = {
            "id": block["id"],
            "type": "table",
            "depth": depth,
            "content": [],
            "metadata": {
                "has_column_header": table_data.get("has_column_header", False),
                "has_row_header": table_data.get("has_row_header", False),
                "table_width": table_data.get("table_width", 0)
            }
        }
        
        # Table content will be populated by table_row blocks
        return parsed_block
    
    def _parse_table_row(self, block: Dict[str, Any], depth: int) -> Dict[str, Any]:
        """Parse a table row block"""
        table_row = block["table_row"]
        cells = table_row.get("cells", [])
        
        parsed_cells = []
        for cell in cells:
            parsed_cells.append(self._extract_rich_text(cell))
        
        return {
            "id": block["id"],
            "type": "table_row",
            "depth": depth,
            "content": parsed_cells,
            "metadata": {}
        }
    
    def _parse_image(self, block: Dict[str, Any], depth: int) -> Dict[str, Any]:
        """Parse an image block"""
        image_data = block["image"]
        image_type = image_data.get("type", "")
        parsed_block = {
            "id": block["id"],
            "type": "image",
            "depth": depth,
            "content": None,
            "metadata": {
                "caption": self._extract_rich_text(image_data.get("caption", [])),
                "type": image_type
            }
        }
        
        # Handle different image types (external vs file)
        if image_type == "external":
            parsed_block["metadata"]["url"] = image_data["external"]["url"]
        elif image_type == "file":
            parsed_block["metadata"]["url"] = image_data["file"]["url"]
            # Download and OCR image if needed
            if self.ocr_enabled:
                try:
                    image_text = self._ocr_image(image_data["file"]["url"])
                    if image_text:
                        parsed_block["content"] = image_text
                except Exception as e:
                    logger.error(f"Failed to OCR image: {e}")
        
        self.embedded_files_count += 1
        return parsed_block
    
    def _parse_file(self, block: Dict[str, Any], depth: int) -> Dict[str, Any]:
        """Parse a file block"""
        file_data = block["file"]
        file_type = file_data.get("type", "")
        parsed_block = {
            "id": block["id"],
            "type": "file",
            "depth": depth,
            "content": None,
            "metadata": {
                "caption": self._extract_rich_text(file_data.get("caption", [])),
                "type": file_type
            }
        }
        
        if file_type == "external":
            parsed_block["metadata"]["url"] = file_data["external"]["url"]
        elif file_type == "file":
            parsed_block["metadata"]["url"] = file_data["file"]["url"]
            # Extract text from file if possible
            try:
                file_text = self._extract_file_text(file_data["file"]["url"])
                if file_text:
                    parsed_block["content"] = file_text
            except Exception as e:
                logger.error(f"Failed to extract text from file: {e}")
        
        self.embedded_files_count += 1
        return parsed_block
    
    def _parse_pdf(self, block: Dict[str, Any], depth: int) -> Dict[str, Any]:
        """Parse a PDF block"""
        pdf_data = block["pdf"]
        pdf_type = pdf_data.get("type", "")
        parsed_block = {
            "id": block["id"],
            "type": "pdf",
            "depth": depth,
            "content": None,
            "metadata": {
                "caption": self._extract_rich_text(pdf_data.get("caption", [])),
                "type": pdf_type
            }
        }
        
        if pdf_type == "external":
            parsed_block["metadata"]["url"] = pdf_data["external"]["url"]
        elif pdf_type == "file":
            parsed_block["metadata"]["url"] = pdf_data["file"]["url"]
            # Extract text from PDF
            try:
                pdf_text = self._extract_pdf_text(pdf_data["file"]["url"])
                if pdf_text:
                    parsed_block["content"] = pdf_text
            except Exception as e:
                logger.error(f"Failed to extract text from PDF: {e}")
        
        self.embedded_files_count += 1
        return parsed_block
    
    def _parse_video(self, block: Dict[str, Any], depth: int) -> Dict[str, Any]:
        """Parse a video block"""
        video_data = block["video"]
        video_type = video_data.get("type", "")
        parsed_block = {
            "id": block["id"],
            "type": "video",
            "depth": depth,
            "content": None,
            "metadata": {
                "caption": self._extract_rich_text(video_data.get("caption", [])),
                "type": video_type
            }
        }
        
        if video_type == "external":
            parsed_block["metadata"]["url"] = video_data["external"]["url"]
        elif video_type == "file":
            parsed_block["metadata"]["url"] = video_data["file"]["url"]
        
        self.embedded_files_count += 1
        return parsed_block
    
    def _parse_bookmark(self, block: Dict[str, Any], depth: int) -> Dict[str, Any]:
        """Parse a bookmark block"""
        bookmark_data = block["bookmark"]
        return {
            "id": block["id"],
            "type": "bookmark",
            "depth": depth,
            "content": self._extract_rich_text(bookmark_data.get("caption", [])),
            "metadata": {
                "url": bookmark_data.get("url", "")
            }
        }
    
    def _parse_embed(self, block: Dict[str, Any], depth: int) -> Dict[str, Any]:
        """Parse an embed block"""
        embed_data = block["embed"]
        return {
            "id": block["id"],
            "type": "embed",
            "depth": depth,
            "content": None,
            "metadata": {
                "url": embed_data.get("url", "")
            }
        }
    
    def _ocr_image(self, image_url: str) -> Optional[str]:
        """Perform OCR on an image to extract text"""
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            
            # Preprocess image for better OCR results
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            text = pytesseract.image_to_string(image)
            return text.strip() if text else None
        except Exception as e:
            logger.error(f"OCR failed for image {image_url}: {e}")
            return None
    
    def _extract_file_text(self, file_url: str) -> Optional[str]:
        """Extract text from various file types"""
        try:
            response = requests.get(file_url, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            # Handle different file types
            if 'text/plain' in content_type:
                return response.text
            elif 'csv' in content_type or 'text/csv' in content_type:
                # Parse CSV
                csv_text = ""
                decoded_content = response.content.decode('utf-8')
                csv_reader = csv.reader(decoded_content.splitlines())
                for row in csv_reader:
                    csv_text += ", ".join(row) + "\n"
                return csv_text
            elif 'application/json' in content_type:
                # Parse JSON
                try:
                    json_data = response.json()
                    return json.dumps(json_data, indent=2)
                except:
                    return response.text
            # Add more file type handlers as needed
            
            return None
        except Exception as e:
            logger.error(f"Failed to extract text from file {file_url}: {e}")
            return None
    
    def _extract_pdf_text(self, pdf_url: str) -> Optional[str]:
        """Extract text from a PDF file"""
        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            
            return text.strip() if text else None
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {pdf_url}: {e}")
            return None
    
    def to_langchain_documents(self, parsed_data: Dict[str, Any]) -> List[LangchainDocument]:
        """Convert parsed Notion data to LangChain documents for AI workflows"""
        documents = []
        
        # Extract main content
        main_content = self._flatten_content(parsed_data["content"])
        if main_content:
            documents.append(LangchainDocument(
                page_content=main_content,
                metadata={
                    "source": parsed_data["url"],
                    "title": parsed_data["title"],
                    "type": parsed_data["type"],
                    "id": parsed_data["id"],
                    "created_time": parsed_data["created_time"],
                    "last_edited_time": parsed_data["last_edited_time"]
                }
            ))
        
        # Add embedded files as separate documents
        embedded_files = self._extract_embedded_files(parsed_data["content"])
        for file_content, file_metadata in embedded_files:
            documents.append(LangchainDocument(
                page_content=file_content,
                metadata=file_metadata
            ))
        
        return documents
    
    def _flatten_content(self, blocks: List[Dict[str, Any]], level: int = 0) -> str:
        """Convert nested block structure to flat text with hierarchy indicators"""
        text_parts = []
        
        for block in blocks:
            # Add appropriate indentation based on depth
            indent = "  " * block.get("depth", 0)
            
            # Handle different block types
            if block["type"].startswith("heading_"):
                level = int(block["type"].split("_")[1])
                heading_prefix = "#" * level + " "
                if block["content"]:
                    text_parts.append(f"{indent}{heading_prefix}{block['content']}\n")
            
            elif block["type"] in ["paragraph", "quote"]:
                if block["content"]:
                    text_parts.append(f"{indent}{block['content']}\n")
            
            elif block["type"] == "code":
                if block["content"]:
                    language = block["metadata"].get("language", "")
                    text_parts.append(f"{indent}```{language}\n{block['content']}\n```\n")
            
            elif block["type"] == "bulleted_list_item":
                if block["content"]:
                    text_parts.append(f"{indent}* {block['content']}\n")
            
            elif block["type"] == "numbered_list_item":
                if block["content"]:
                    text_parts.append(f"{indent}1. {block['content']}\n")
            
            elif block["type"] == "to_do":
                if block["content"]:
                    checkbox = "[x]" if block["metadata"].get("checked") else "[ ]"
                    text_parts.append(f"{indent}{checkbox} {block['content']}\n")
            
            elif block["type"] == "table":
                # Process table rows
                if "children" in block:
                    for row in block["children"]:
                        if row["type"] == "table_row" and row["content"]:
                            text_parts.append(f"{indent}| {' | '.join(row['content'])} |\n")
            
            elif block["type"] == "image" and block["content"]:
                text_parts.append(f"{indent}[Image: {block['content']}]\n")
            
            elif block["type"] in ["file", "pdf"] and block["content"]:
                text_parts.append(f"{indent}[{block['type'].upper()}: {block['content']}]\n")
            
            elif block["type"] == "bookmark" and block["content"]:
                text_parts.append(f"{indent}[Bookmark: {block['content']}]({block['metadata'].get('url', '')})\n")
            
            elif block["type"] == "embed":
                text_parts.append(f"{indent}[Embed: {block['metadata'].get('url', '')}]\n")
            
            elif block["type"] == "equation" and block["content"]:
                text_parts.append(f"{indent}Equation: {block['content']}\n")
            
            elif block["type"] == "divider":
                text_parts.append(f"{indent}---\n")
            
            elif block["type"] == "table_of_contents":
                text_parts.append(f"{indent}Table of Contents\n")
            
            elif block["type"] == "child_page" and block["content"]:
                text_parts.append(f"{indent}Child Page: {block['content']}\n")
            
            elif block["type"] == "child_database" and block["content"]:
                text_parts.append(f"{indent}Database: {block['content']}\n")
            
            # Recursively process children
            if "children" in block:
                children_text = self._flatten_content(block["children"], level + 1)
                text_parts.append(children_text)
        
        return "".join(text_parts)
    
    def _extract_embedded_files(self, blocks: List[Dict[str, Any]]) -> List[tuple]:
        """Extract embedded files as separate content pieces"""
        embedded_files = []
        
        for block in blocks:
            if block["type"] in ["image", "file", "pdf"] and block["content"]:
                embedded_files.append((
                    block["content"],
                    {
                        "source_type": block["type"],
                        "caption": block["metadata"].get("caption", ""),
                        "url": block["metadata"].get("url", ""),
                        "block_id": block["id"]
                    }
                ))
            
            # Recursively check children
            if "children" in block:
                embedded_files.extend(self._extract_embedded_files(block["children"]))
        
        return embedded_files