import json
import re
from typing import Dict, Any, Set, Tuple

# --- CONFIGURATION ---
INPUT_PATH = "model-2/data/basic_data_3.jsonl"
OUTPUT_PATH = "model-2/data/basic_data_3.cleaned.jsonl"
COLUMNS_TO_DROP = ["description", "mitigation"]
BROAD_CATEGORY_FIELD = "cwe_category"
CWE_ID_FIELD = "cwe_id" # New field for the specific CWE ID

# --- VULNERABILITY MAPPING (CWE ID and Broad Category) ---
# Maps specific vulnerability types to a tuple: (CWE_ID, Broad_Category)
VULNERABILITY_MAPPING: Dict[str, Tuple[str, str]] = {
    # --------------------------------------------------------------------------------
    # 1. INJECTION & INPUT VALIDATION (CWE-89, CWE-79, CWE-77)
    # --------------------------------------------------------------------------------
    'SQL Injection': ('CWE-89', 'Injection & Input Flaws'), 'Command Injection': ('CWE-77', 'Injection & Input Flaws'),
    'OS Command Injection': ('CWE-78', 'Injection & Input Flaws'), 'LDAP Injection': ('CWE-90', 'Injection & Input Flaws'),
    'NoSQL Injection': ('CWE-89', 'Injection & Input Flaws'), 'GraphQL Injection': ('CWE-89', 'Injection & Input Flaws'),
    'Template Injection': ('CWE-94', 'Injection & Input Flaws'), 'Insecure Template Injection': ('CWE-94', 'Injection & Input Flaws'),
    'Server-Side Template Injection': ('CWE-94', 'Injection & Input Flaws'), 'Insecure Regex DoS': ('CWE-1333', 'Injection & Input Flaws'),
    'Insecure Regular Expression': ('CWE-1333', 'Injection & Input Flaws'), 'Regex Injection': ('CWE-1333', 'Injection & Input Flaws'),
    'Insecure Eval': ('CWE-94', 'Injection & Input Flaws'), 'HTTP Parameter Pollution': ('CWE-20', 'Injection & Input Flaws'),
    'Script Engine Injection': ('CWE-94', 'Injection & Input Flaws'), 'Code Injection via Dynamic Execution': ('CWE-94', 'Injection & Input Flaws'),
    'Dynamic Code Execution': ('CWE-94', 'Injection & Input Flaws'), 'Dynamic Code Evaluation': ('CWE-94', 'Injection & Input Flaws'),
    'Dynamic Function Call': ('CWE-94', 'Injection & Input Flaws'), 'Dynamic Include': ('CWE-98', 'Injection & Input Flaws'),
    'Reflection Injection': ('CWE-74', 'Injection & Input Flaws'), 'SQL Injection via Dynamic Queries': ('CWE-89', 'Injection & Input Flaws'),
    'Unvalidated Input in GraphQL': ('CWE-20', 'Injection & Input Flaws'), 'Cross-Site Scripting': ('CWE-79', 'Injection & Input Flaws'),

    # --------------------------------------------------------------------------------
    # 2. DESERIALIZATION, XML, & DATA HANDLING (CWE-502, CWE-611)
    # --------------------------------------------------------------------------------
    'Deserialization': ('CWE-502', 'Data Deserialization & XXE'), 'Insecure Deserialization': ('CWE-502', 'Data Deserialization & XXE'),
    'Insecure Deserialization (JSON)': ('CWE-502', 'Data Deserialization & XXE'), 'YAML Deserialization': ('CWE-502', 'Data Deserialization & XXE'),
    'Insecure YAML Deserialization': ('CWE-502', 'Data Deserialization & XXE'), 'Insecure XML Deserialization': ('CWE-502', 'Data Deserialization & XXE'),
    'XXE': ('CWE-611', 'Data Deserialization & XXE'), 'XXE (XML External Entity)': ('CWE-611', 'Data Deserialization & XXE'),
    'Insecure Pickle Usage': ('CWE-502', 'Data Deserialization & XXE'), 'JNDI Injection': ('CWE-917', 'Data Deserialization & XXE'),

    # --------------------------------------------------------------------------------
    # 3. ACCESS CONTROL & FILE HANDLING (CWE-287, CWE-22)
    # --------------------------------------------------------------------------------
    'Broken Authentication': ('CWE-287', 'Access & Authorization'), 'Broken Access Control': ('CWE-285', 'Access & Authorization'),
    'IDOR': ('CWE-639', 'Access & Authorization'), 'Insecure Direct Object Reference': ('CWE-639', 'Access & Authorization'),
    'Mass Assignment': ('CWE-915', 'Access & Authorization'), 'Mass Assignment Vulnerability': ('CWE-915', 'Access & Authorization'),
    'Unrestricted File Upload': ('CWE-434', 'Access & Authorization'), 'Insecure File Upload': ('CWE-434', 'Access & Authorization'),
    'Insecure File Access': ('CWE-732', 'Access & Authorization'), 'Directory Traversal': ('CWE-22', 'Access & Authorization'),
    'Path Traversal': ('CWE-22', 'Access & Authorization'), 'File Inclusion': ('CWE-98', 'Access & Authorization'),
    'Remote File Inclusion': ('CWE-98', 'Access & Authorization'), 'Clickjacking': ('CWE-1021', 'Access & Authorization'),
    'Open Redirect': ('CWE-601', 'Access & Authorization'), 'Insecure Redirect': ('CWE-601', 'Access & Authorization'),
    'Insecure Redirects': ('CWE-601', 'Access & Authorization'), 'Insecure OAuth Redirect': ('CWE-601', 'Access & Authorization'),
    'Insecure File Download': ('CWE-22', 'Access & Authorization'),

    # --------------------------------------------------------------------------------
    # 4. CRYPTO & SECRETS (CWE-327, CWE-330)
    # --------------------------------------------------------------------------------
    'Insecure Password Storage': ('CWE-257', 'Cryptography & Secrets'), 'Weak Password Hashing': ('CWE-916', 'Cryptography & Secrets'),
    'Hardcoded Credentials': ('CWE-798', 'Cryptography & Secrets'), 'Hardcoded Secrets': ('CWE-798', 'Cryptography & Secrets'),
    'Insecure Randomness': ('CWE-338', 'Cryptography & Secrets'), 'Insecure Random': ('CWE-338', 'Cryptography & Secrets'),
    'Weak Random Number Generation': ('CWE-330', 'Cryptography & Secrets'), 'Insecure Random Number Generation': ('CWE-338', 'Cryptography & Secrets'),
    'Insecure Random Token Lifetime': ('CWE-338', 'Cryptography & Secrets'), 'Insecure API Key Exposure': ('CWE-200', 'Cryptography & Secrets'),

    # --------------------------------------------------------------------------------
    # 5. SESSION & TOKEN MANAGEMENT (CWE-613, CWE-384)
    # --------------------------------------------------------------------------------
    'Insecure Session Handling': ('CWE-613', 'Session & Token Management'), 'Session Fixation': ('CWE-384', 'Session & Token Management'),
    'Insecure Session Cookies': ('CWE-614', 'Session & Token Management'), 'Insecure Session Storage': ('CWE-614', 'Session & Token Management'),
    'Insecure Session Timeout': ('CWE-613', 'Session & Token Management'), 'Insecure Session': ('CWE-613', 'Session & Token Management'),
    'Insecure Session Regeneration': ('CWE-613', 'Session & Token Management'), 'Insecure Session ID Generation': ('CWE-330', 'Session & Token Management'),
    'Insecure JWT Handling': ('CWE-287', 'Session & Token Management'), 'Insecure JWT': ('CWE-287', 'Session & Token Management'),
    'JWT None Algorithm': ('CWE-347', 'Session & Token Management'), 'Insecure Token Validation': ('CWE-347', 'Session & Token Management'),
    'Insecure Cookie Scope': ('CWE-614', 'Session & Token Management'), 'Insecure Cookie Configuration': ('CWE-614', 'Session & Token Management'),
    'Insecure Cookie Flags': ('CWE-614', 'Session & Token Management'), 'Insecure Cookies': ('CWE-614', 'Session & Token Management'),
    'Insecure CSRF Token Storage': ('CWE-352', 'Session & Token Management'), 'Insecure Password Reset Token': ('CWE-640', 'Session & Token Management'),

    # --------------------------------------------------------------------------------
    # 6. CONFIGURATION & ENVIRONMENT (CWE-16, CWE-209)
    # --------------------------------------------------------------------------------
    'CORS Misconfig': ('CWE-346', 'Configuration & Environment'), 'CORS Misconfiguration': ('CWE-346', 'Configuration & Environment'),
    'Insecure CORS': ('CWE-346', 'Configuration & Environment'), 'Insecure CORS Configuration': ('CWE-346', 'Configuration & Environment'),
    'Insecure HTTP Methods': ('CWE-749', 'Configuration & Environment'), 'Insecure HTTP Method Handling': ('CWE-749', 'Configuration & Environment'),
    'Insecure HTTP Headers': ('CWE-16', 'Configuration & Environment'), 'Improper Error Handling': ('CWE-209', 'Configuration & Environment'),
    'Insecure Error Handling': ('CWE-209', 'Configuration & Environment'), 'Insecure Content-Type Handling': ('CWE-433', 'Configuration & Environment'),
    'Insecure Content Security Policy': ('CWE-16', 'Configuration & Environment'), 'Insecure HTTP Client': ('CWE-939', 'Configuration & Environment'),
    'Insecure TLS Configuration': ('CWE-326', 'Configuration & Environment'), 'Insecure File Permissions': ('CWE-732', 'Configuration & Environment'),
    'Insecure Logging': ('CWE-532', 'Configuration & Environment'), 'Insecure GraphQL Introspection': ('CWE-200', 'Configuration & Environment'),
    'Insecure Cache Control': ('CWE-524', 'Configuration & Environment'),

    # --------------------------------------------------------------------------------
    # 7. SERVER/CLIENT REQUEST FORGERY (CWE-352, CWE-918)
    # --------------------------------------------------------------------------------
    'CSRF': ('CWE-352', 'Server/Client Request Forgery'), 'Insecure CSRF Token Handling': ('CWE-352', 'Server/Client Request Forgery'),
    'SSRF': ('CWE-918', 'Server/Client Request Forgery'), 'Server-Side Request Forgery': ('CWE-918', 'Server/Client Request Forgery'),
    'Server-Side Request Forgery (SSRF)': ('CWE-918', 'Server/Client Request Forgery'),

    # --------------------------------------------------------------------------------
    # 8. APPLICATION LOGIC & OTHERS (CWE-693)
    # --------------------------------------------------------------------------------
    'Log4Shell': ('CWE-502', 'Application Logic & Others'), 'Zip Slip': ('CWE-22', 'Application Logic & Others'),
    'GraphQL Depth Limit': ('CWE-400', 'Application Logic & Others'), 'Race Condition': ('CWE-362', 'Application Logic & Others'),
    'Message Passing Race': ('CWE-362', 'Application Logic & Others'), 'Prototype Pollution': ('CWE-1321', 'Application Logic & Others'),
    'Prototype Pollution Equivalent': ('CWE-1321', 'Application Logic & Others'), 'Prototype Pollution in Dependencies': ('CWE-1321', 'Application Logic & Others'),
    'Dynamic Constant Assignment': ('CWE-94', 'Application Logic & Others'), 'Dynamic Method Invocation': ('CWE-94', 'Application Logic & Others'),
    'Insecure Output Encoding': ('CWE-116', 'Application Logic & Others'), 'Insecure Dependency': ('CWE-1104', 'Application Logic & Others'),
    'Insecure Dependency Management': ('CWE-1104', 'Application Logic & Others'), 'Rate Limiting Bypass': ('CWE-770', 'Application Logic & Others'),
    'Insecure Rate Limiting': ('CWE-770', 'Application Logic & Others'), 'Insecure Password Reset': ('CWE-640', 'Application Logic & Others'),
    'Insecure WebSocket': ('CWE-346', 'Application Logic & Others'), 'Insecure JSONP': ('CWE-346', 'Application Logic & Others'),
    'Unsafe Memory Access': ('CWE-119', 'Application Logic & Others'),
}

def get_cwe_and_category(vulnerability_type: str) -> Tuple[str, str, str]:
    """
    Maps a fine-grained vulnerability type to its CWE ID and broad category.
    Returns ("N/A", "Uncategorized/Unknown") if no mapping is found.
    """
    mapping = VULNERABILITY_MAPPING.get(vulnerability_type)
    if mapping:
        return mapping[0], mapping[1], mapping[1] # cwe_id, broad_category, broad_category_for_check
    else:
        return "N/A", "Uncategorized/Unknown", "Uncategorized/Unknown"

def iter_objects(raw: str):
    """
    Yields JSON object strings from concatenated objects split on '}\n{',
    handling variations in whitespace and newlines.
    """
    # Modified split to be less aggressive with inner content but still handle the line separation
    parts = re.split(r"}\s*\n+\s*{", raw.strip())
    for p in parts:
        if not p.startswith("{"):
            p = "{" + p
        if not p.endswith("}"):
            p = p + "}"
        yield p

def main():
    """
    Runs the complete data processing pipeline: loads, cleans, deduplicates, 
    aggregates (categorizes with CWE ID), and saves the final dataset.
    """
    try:
        with open(INPUT_PATH, "r", encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at '{INPUT_PATH}'. Please check the file path.")
        return

    # Initialize tracking variables
    seen: Set[str] = set()
    decode_errors = 0
    dup_skipped = 0
    written = 0
    total_parsed = 0
    uncategorized_types: Set[str] = set()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for total_parsed, obj_str in enumerate(iter_objects(raw), start=1):
            try:
                rec: Dict[str, Any] = json.loads(obj_str)
            except json.JSONDecodeError:
                decode_errors += 1
                continue

            # DEDUPLICATION (using the 'id' field)
            id_val = rec.get("id")
            if isinstance(id_val, str):
                # Check for both 'id' and 'code_snippet' for more robust deduplication
                unique_key = f"{id_val}:{rec.get('code_snippet', '')}"
                if unique_key in seen:
                    dup_skipped += 1
                    continue
                seen.add(unique_key)
            
            # DROP UNNECESSARY COLUMNS
            for col in COLUMNS_TO_DROP:
                rec.pop(col, None) 
            
            # AGGREGATION / CATEGORIZATION (now includes CWE ID)
            fine_grained_type = rec.get("vulnerability_type", "N/A")
            
            cwe_id, broad_category, category_check = get_cwe_and_category(fine_grained_type)
            
            rec[CWE_ID_FIELD] = cwe_id # ADD CWE ID
            rec[BROAD_CATEGORY_FIELD] = broad_category # ADD Broad Category
            
            if category_check == "Uncategorized/Unknown" and fine_grained_type != "N/A":
                uncategorized_types.add(fine_grained_type)

            # WRITE CLEAN, AGGREGATED RECORD
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print("--- Data Processing Summary ---")
    print(f"Input file: {INPUT_PATH}")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Total objects parsed from input: {total_parsed}")
    print(f"JSON decode errors skipped: {decode_errors}")
    print(f"Duplicate IDs skipped: {dup_skipped}")
    print(f"Records successfully cleaned & aggregated: {written}")
    
    if uncategorized_types:
        print(f"\n⚠️ WARNING: {len(uncategorized_types)} types were UNCATEGORIZED.")
        print("Please review and add the following to VULNERABILITY_MAPPING:")
        for utype in sorted(list(uncategorized_types)):
            print(f"  - {utype}")
    else:
        print("\n✅ All records were successfully mapped to one of the broad categories.")

if __name__ == "__main__":
    main()
