from typing import List, Dict, Any

def generate_markdown_table(data: List[Dict[str, Any]]) -> str:
    """
    Converts a list of dictionaries into a clean Markdown table.
    """
    if not data:
        return ""

    # Extract headers from the first dictionary keys
    headers = list(data[0].keys())
    
    # Capitalize headers for display
    display_headers = [h.replace("_", " ").title() for h in headers]
    
    # Create the header row
    header_row = "| " + " | ".join(display_headers) + " |"
    # Create the separator row
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    # Create data rows
    body_rows = []
    for row in data:
        row_values = [str(row.get(h, "")).replace("|", "\\|") for h in headers]
        body_rows.append("| " + " | ".join(row_values) + " |")
    
    return "\n".join([header_row, separator_row] + body_rows)
