import json
import re
from typing import Dict, Any, Set

# --- CONFIGURATION ---
INPUT_PATH = "basic_data_3.jsonl" 
OUTPUT_PATH = "basic_data_3.aggregated.jsonl" 
COLUMNS_TO_DROP = ["description", "mitigation"] 
BROAD_CATEGORY_FIELD = "cwe_category"

# --- VULNERABILITY MAPPING (Consolidated and Expanded) ---
# Maps specific vulnerability types to a broad, machine-learnable category (approx. 10 CWE classes).
VULNERABILITY_MAPPING = {
    # --------------------------------------------------------------------------------
    # 1. INJECTION & INPUT VALIDATION (CWE-89, CWE-79, CWE-77)
    # --------------------------------------------------------------------------------
    'SQL Injection': 'Injection & Input Flaws', 'Command Injection': 'Injection & Input Flaws',
    'OS Command Injection': 'Injection & Input Flaws', 'LDAP Injection': 'Injection & Input Flaws',
    'NoSQL Injection': 'Injection & Input Flaws', 'GraphQL Injection': 'Injection & Input Flaws',
    'Template Injection': 'Injection & Input Flaws', 'Insecure Template Injection': 'Injection & Input Flaws',
    'Server-Side Template Injection': 'Injection & Input Flaws', 'Insecure Regex DoS': 'Injection & Input Flaws',
    'Insecure Regular Expression': 'Injection & Input Flaws', 'Regex Injection': 'Injection & Input Flaws',
    'Insecure Eval': 'Injection & Input Flaws', 'HTTP Parameter Pollution': 'Injection & Input Flaws',
    'Script Engine Injection': 'Injection & Input Flaws', 'Code Injection via Dynamic Execution': 'Injection & Input Flaws',
    'Dynamic Code Execution': 'Injection & Input Flaws', 'Dynamic Code Evaluation': 'Injection & Input Flaws',
    'Dynamic Function Call': 'Injection & Input Flaws', 'Dynamic Include': 'Injection & Input Flaws',
    'Reflection Injection': 'Injection & Input Flaws', 'SQL Injection via Dynamic Queries': 'Injection & Input Flaws',
    'Unvalidated Input in GraphQL': 'Injection & Input Flaws',

    # --------------------------------------------------------------------------------
    # 2. DESERIALIZATION, XML, & DATA HANDLING (CWE-502, CWE-611)
    # --------------------------------------------------------------------------------
    'Deserialization': 'Data Deserialization & XXE', 'Insecure Deserialization': 'Data Deserialization & XXE',
    'Insecure Deserialization (JSON)': 'Data Deserialization & XXE', 'YAML Deserialization': 'Data Deserialization & XXE',
    'Insecure YAML Deserialization': 'Data Deserialization & XXE', 'Insecure XML Deserialization': 'Data Deserialization & XXE',
    'XXE': 'Data Deserialization & XXE', 'XXE (XML External Entity)': 'Data Deserialization & XXE',
    'Insecure Pickle Usage': 'Data Deserialization & XXE', 'JNDI Injection': 'Data Deserialization & XXE',

    # --------------------------------------------------------------------------------
    # 3. ACCESS CONTROL & FILE HANDLING (CWE-287, CWE-22)
    # --------------------------------------------------------------------------------
    'Broken Authentication': 'Access & Authorization', 'Broken Access Control': 'Access & Authorization',
    'IDOR': 'Access & Authorization', 'Insecure Direct Object Reference': 'Access & Authorization',
    'Mass Assignment': 'Access & Authorization', 'Mass Assignment Vulnerability': 'Access & Authorization',
    'Unrestricted File Upload': 'Access & Authorization', 'Insecure File Upload': 'Access & Authorization',
    'Insecure File Access': 'Access & Authorization', 'Directory Traversal': 'Access & Authorization',
    'Path Traversal': 'Access & Authorization', 'File Inclusion': 'Access & Authorization',
    'Remote File Inclusion': 'Access & Authorization', 'Clickjacking': 'Access & Authorization',
    'Open Redirect': 'Access & Authorization', 'Insecure Redirect': 'Access & Authorization',
    'Insecure Redirects': 'Access & Authorization', 'Insecure OAuth Redirect': 'Access & Authorization',
    'Insecure File Download': 'Access & Authorization',

    # --------------------------------------------------------------------------------
    # 4. CRYPTO & SECRETS (CWE-327, CWE-330)
    # --------------------------------------------------------------------------------
    'Insecure Password Storage': 'Cryptography & Secrets', 'Weak Password Hashing': 'Cryptography & Secrets',
    'Hardcoded Credentials': 'Cryptography & Secrets', 'Hardcoded Secrets': 'Cryptography & Secrets',
    'Insecure Randomness': 'Cryptography & Secrets', 'Insecure Random': 'Cryptography & Secrets',
    'Weak Random Number Generation': 'Cryptography & Secrets', 'Insecure Random Number Generation': 'Cryptography & Secrets',
    'Insecure Random Token Lifetime': 'Cryptography & Secrets', 'Insecure API Key Exposure': 'Cryptography & Secrets',

    # --------------------------------------------------------------------------------
    # 5. SESSION & TOKEN MANAGEMENT (CWE-613, CWE-384)
    # --------------------------------------------------------------------------------
    'Insecure Session Handling': 'Session & Token Management', 'Session Fixation': 'Session & Token Management',
    'Insecure Session Cookies': 'Session & Token Management', 'Insecure Session Storage': 'Session & Token Management',
    'Insecure Session Timeout': 'Session & Token Management', 'Insecure Session': 'Session & Token Management',
    'Insecure Session Regeneration': 'Session & Token Management', 'Insecure Session ID Generation': 'Session & Token Management',
    'Insecure JWT Handling': 'Session & Token Management', 'Insecure JWT': 'Session & Token Management',
    'JWT None Algorithm': 'Session & Token Management', 'Insecure Token Validation': 'Session & Token Management',
    'Insecure Cookie Scope': 'Session & Token Management', 'Insecure Cookie Configuration': 'Session & Token Management',
    'Insecure Cookie Flags': 'Session & Token Management', 'Insecure Cookies': 'Session & Token Management',
    'Insecure CSRF Token Storage': 'Session & Token Management', 'Insecure Password Reset Token': 'Session & Token Management',

    # --------------------------------------------------------------------------------
    # 6. CONFIGURATION & ENVIRONMENT (CWE-16, CWE-209)
    # --------------------------------------------------------------------------------
    'CORS Misconfig': 'Configuration & Environment', 'CORS Misconfiguration': 'Configuration & Environment', 
    'Insecure CORS': 'Configuration & Environment', 'Insecure CORS Configuration': 'Configuration & Environment',
    'Insecure HTTP Methods': 'Configuration & Environment', 'Insecure HTTP Method Handling': 'Configuration & Environment',
    'Insecure HTTP Headers': 'Configuration & Environment', 'Improper Error Handling': 'Configuration & Environment',
    'Insecure Error Handling': 'Configuration & Environment', 'Insecure Content-Type Handling': 'Configuration & Environment',
    'Insecure Content Security Policy': 'Configuration & Environment', 'Insecure HTTP Client': 'Configuration & Environment',
    'Insecure TLS Configuration': 'Configuration & Environment', 'Insecure File Permissions': 'Configuration & Environment',
    'Insecure Logging': 'Configuration & Environment', 'Insecure GraphQL Introspection': 'Configuration & Environment',
    'Insecure Cache Control': 'Configuration & Environment', 
    
    # --------------------------------------------------------------------------------
    # 7. SERVER/CLIENT REQUEST FORGERY (CWE-352, CWE-918)
    # --------------------------------------------------------------------------------
    'CSRF': 'Server/Client Request Forgery', 'Insecure CSRF Token Handling': 'Server/Client Request Forgery',
    'SSRF': 'Server/Client Request Forgery', 'Server-Side Request Forgery': 'Server/Client Request Forgery',
    'Server-Side Request Forgery (SSRF)': 'Server/Client Request Forgery', 
    
    # --------------------------------------------------------------------------------
    # 8. APPLICATION LOGIC & OTHERS (CWE-693)
    # --------------------------------------------------------------------------------
    'Log4Shell': 'Application Logic & Others', 'Zip Slip': 'Application Logic & Others',
    'GraphQL Depth Limit': 'Application Logic & Others', 'Race Condition': 'Application Logic & Others',
    'Message Passing Race': 'Application Logic & Others', 'Prototype Pollution': 'Application Logic & Others',
    'Prototype Pollution Equivalent': 'Application Logic & Others', 'Prototype Pollution in Dependencies': 'Application Logic & Others', 
    'Dynamic Constant Assignment': 'Application Logic & Others', 'Dynamic Method Invocation': 'Application Logic & Others', 
    'Insecure Output Encoding': 'Application Logic & Others', 'Insecure Dependency': 'Application Logic & Others',
    'Insecure Dependency Management': 'Application Logic & Others', 'Rate Limiting Bypass': 'Application Logic & Others',
    'Insecure Rate Limiting': 'Application Logic & Others', 'Insecure Password Reset': 'Application Logic & Others',
    'Insecure WebSocket': 'Application Logic & Others', 'Insecure JSONP': 'Application Logic & Others', 
    'Unsafe Memory Access': 'Application Logic & Others', 
}


def iter_objects(raw: str):
    """
    Yields JSON object strings from concatenated objects split on '}\n{',
    handling variations in whitespace and newlines.
    """
    parts = re.split(r"}\s*\n*\s*{", raw.strip())
    for p in parts:
        if not p.startswith("{"):
            p = "{" + p
        if not p.endswith("}"):
            p = p + "}"
        yield p

def get_broad_category(vulnerability_type: str) -> str:
    """
    Maps a fine-grained vulnerability type to a broad, standardized category.
    """
    return VULNERABILITY_MAPPING.get(vulnerability_type, "Uncategorized/Unknown")

def main():
    """
    Runs the complete data processing pipeline: loads, cleans, deduplicates, 
    aggregates (categorizes), and saves the final dataset.
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
    total_split = 0
    uncategorized_types: Set[str] = set()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for total_split, obj_str in enumerate(iter_objects(raw), start=1):
            try:
                rec: Dict[str, Any] = json.loads(obj_str)
            except json.JSONDecodeError:
                decode_errors += 1
                continue

            # DEDUPLICATION (using the 'id' field)
            id_val = rec.get("id")
            if isinstance(id_val, str):
                if id_val in seen:
                    dup_skipped += 1
                    continue
                seen.add(id_val)
            
            #  DROP UNNECESSARY COLUMNS
            for col in COLUMNS_TO_DROP:
                rec.pop(col, None) 
            
            # AGGREGATION / CATEGORIZATION
            fine_grained_type = rec.get("vulnerability_type", "N/A")
            broad_category = get_broad_category(fine_grained_type)
            
            rec[BROAD_CATEGORY_FIELD] = broad_category
            
            if broad_category == "Uncategorized/Unknown" and fine_grained_type != "N/A":
                uncategorized_types.add(fine_grained_type)

            # WRITE CLEAN, AGGREGATED RECORD
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print("--- Data Processing Summary ---")
    print(f"Input file: {INPUT_PATH}")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Total objects parsed from input: {total_split}")
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
