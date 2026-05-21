# ### --- [IMPORTS] --- ###

from typing import List, Dict, Any


# ### --- [TABLE FORMATTING UTILITIES] --- ###

# [Elaborative Breakdown]
# Markdown Grid Formatting and Sanitization:
# When outputting database rows (raw SQL query results) directly to the user in a chatbot
# interface, we must present them in a visually premium, structured format. Markdown tables
# are the industry standard for this.
#
# However, if any database column value contains the pipe character (`|`), standard markdown
# table parsers in UI frontends will misinterpret it as a column delimiter, resulting in broken,
# misaligned grids. 
#
# We solve this by implementing a fast, dynamic, one-pass string escaping filter:
# 1. We dynamically extract header keys from the first dictionary element to lock in column counts.
# 2. We normalize and sanitize all column headers to reader-friendly title-case configurations.
# 3. We iterate over rows, transforming database fields into string representations and surgically
#    replacing literal pipe characters (`|`) with standard escaped equivalents (`\|`), preventing 
#    rendering breakage while keeping the transformation overhead near O(N * M) where N is row count
#    and M is column count.
def generate_markdown_table(data: List[Dict[str, Any]]) -> str:
    """Converts a list of row dictionaries into a clean, readable Markdown table.

    Dynamically extracts columns from key sets, cleanses special characters 
    (such as replacing pipe symbols `|` with backslash-escapes to prevent 
    Markdown rendering breakage), and builds clean aligned column grids.

    Args:
        data (List[Dict[str, Any]]): A list of row dictionary mapping column names to values.

    Returns:
        str: Renders output as formatted GitHub-Flavored Markdown table text.
    """
    if not data:
        # Zero-check: If data array is empty, return empty string to avoid index out of bounds.
        return ""

    # 1. Extract headers from the first dictionary keys to lock down column structure.
    headers: List[str] = list(data[0].keys())
    
    # 2. Pretty-format headers for a premium look: replace underscores and apply title casing.
    display_headers: List[str] = [h.replace("_", " ").title() for h in headers]
    
    # 3. Build GFM header row and the underlying alignment dividers (defaulting to standard left-alignment).
    header_row: str = "| " + " | ".join(display_headers) + " |"
    separator_row: str = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    # 4. Process data rows contextually escaping any markdown pipe separators to preserve alignment.
    body_rows: List[str] = []
    for row in data:
        # Extract, stringify, and escape the pipe symbol to defend against rendering glitches.
        row_values: List[str] = [str(row.get(h, "")).replace("|", "\\|") for h in headers]
        body_rows.append("| " + " | ".join(row_values) + " |")
    
    # 5. Stitch header, separator, and data rows into a single multi-line string.
    return "\n".join([header_row, separator_row] + body_rows)


