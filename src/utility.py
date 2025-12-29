def get_document_type(file_name: str) -> str:
    name_lower = file_name.lower()
    
    # Comprehensive Keyword Mapping
    keyword_library = {
        "technical": [
            "spec", "api", "manual", "guide", "tech", "readme", "architecture", 
            "deployment", "schema", "codebase", "integration", "config", "sysadmin",
            "developer", "documentation", "blueprint", "setup", "install","compute","computing","intelligence"
        ],
        "legal": [
            "contract", "policy", "terms", "agreement", "nda", "compliance", 
            "clause", "liability", "litigation", "statute", "regulation", 
            "amendment", "dpa", "privacy", "tos", "disclaimer", "legal"
        ],
        "financial": [
            "report", "invoice", "billing", "budget", "quarterly", "revenue", 
            "fiscal", "audit", "tax", "statement", "ledger", "payroll", 
            "forecast", "expense", "profit", "loss", "ebitda", "transaction"
        ],
        "hr": [
            "resume", "onboarding", "performance", "benefits", "payroll", 
            "hiring", "job-description", "recruitment", "employee", "handbook", 
            "training", "eval", "interview", "vacation", "culture"
        ],
        "medical": [
            "bio", "patient", "clinical", "health", "history", "prescription", 
            "treatment", "diagnosis", "chart", "lab-result", "ehr", 
            "medical", "symptoms", "dosage", "pharmaceutical"
        ]
    }
    
    priority_order = ["technical","legal", "financial", "hr", "medical"]
    
    # Using the same library as above
    results = {}
    for doc_type, keywords in keyword_library.items():
        if any(kw in name_lower for kw in keywords):
            results[doc_type] = True
            
    for p_type in priority_order:
        if p_type in results:
            return p_type
            
    return "general"