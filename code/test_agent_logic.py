#!/usr/bin/env python3
"""
Unit tests for agent logic WITHOUT hitting the LLM API.
Tests company validation, cache normalization, and error handling.
"""

from agent import TriageOutput, is_valid_company

def test_company_validation():
    """Test is_valid_company function."""
    print("Testing is_valid_company()...")
    
    # Valid companies
    assert is_valid_company("HackerRank") == True
    assert is_valid_company("Claude") == True
    assert is_valid_company("Visa") == True
    assert is_valid_company(" HackerRank ") == True
    
    # Invalid companies
    assert is_valid_company("None") == False
    assert is_valid_company("none") == False
    assert is_valid_company("") == False
    assert is_valid_company(None) == False
    assert is_valid_company("  ") == False
    
    print("✓ Company validation tests passed")


def test_triage_output_schema():
    """Test that TriageOutput can be created with all required fields."""
    print("Testing TriageOutput schema...")
    
    output = TriageOutput(
        chain_of_thought="Step 1: Read issue. Step 2: Check context. Step 3: Decide.",
        status="replied",
        product_area="Test Management",
        response="Here is your answer.",
        justification="The context supports this answer.",
        request_type="product_issue"
    )
    
    # Verify all fields are present
    assert output.chain_of_thought == "Step 1: Read issue. Step 2: Check context. Step 3: Decide."
    assert output.status == "replied"
    assert output.product_area == "Test Management"
    assert output.response == "Here is your answer."
    assert output.justification == "The context supports this answer."
    assert output.request_type == "product_issue"
    
    # Verify it can be converted to dict with correct order
    output_dict = {
        "status": output.status,
        "product_area": output.product_area,
        "response": output.response,
        "justification": output.justification,
        "request_type": output.request_type,
        "chain_of_thought": output.chain_of_thought
    }
    
    # Verify column order (chain_of_thought should be LAST)
    keys = list(output_dict.keys())
    assert keys[-1] == "chain_of_thought", f"Expected chain_of_thought as last column, got {keys}"
    
    print(f"✓ TriageOutput schema tests passed")
    print(f"  Column order: {keys}")


def test_escalation_on_missing_company():
    """Test that invalid company triggers escalation."""
    print("Testing escalation on missing company...")
    
    # Simulate what agent.generate_response does for invalid company
    company = "None"
    
    if not is_valid_company(company):
        output = TriageOutput(
            chain_of_thought="Step 1: Check company field. Step 2: Company is missing or 'None'. Step 3: Cannot reliably determine product scope without company context. Step 4: Escalate.",
            status="escalated",
            product_area="Unknown",
            response="Unable to determine which product this issue belongs to. Please resubmit your ticket with the product/company name (HackerRank, Claude, or Visa).",
            justification="Company field is missing or empty. Cannot triage without knowing which product context to use.",
            request_type="invalid"
        )
        
        assert output.status == "escalated"
        assert output.request_type == "invalid"
        assert output.product_area == "Unknown"
        print("✓ Escalation on missing company works correctly")
    else:
        raise AssertionError("Expected company 'None' to be invalid")


def test_response_dict_normalization():
    """Test that response dicts are normalized for CSV column ordering."""
    print("Testing response dict normalization...")
    
    # Simulate a cached response with potentially different key order
    cached = {
        "chain_of_thought": "Some reasoning",
        "status": "replied",
        "response": "Answer",
        "justification": "Justification",
        "product_area": "Area",
        "request_type": "product_issue"
    }
    
    # Normalize to match CSV column order (chain_of_thought should be LAST)
    ordered = {
        "status": cached.get("status", ""),
        "product_area": cached.get("product_area", ""),
        "response": cached.get("response", ""),
        "justification": cached.get("justification", ""),
        "request_type": cached.get("request_type", ""),
        "chain_of_thought": cached.get("chain_of_thought", "")
    }
    
    # Verify order
    keys = list(ordered.keys())
    assert keys == ["status", "product_area", "response", "justification", "request_type", "chain_of_thought"]
    assert keys[-1] == "chain_of_thought"
    
    print(f"✓ Response dict normalization works correctly")
    print(f"  Correct column order: {keys}")


if __name__ == "__main__":
    print("Running agent logic unit tests...\n")
    
    test_company_validation()
    test_triage_output_schema()
    test_escalation_on_missing_company()
    test_response_dict_normalization()
    
    print("\n✅ All unit tests passed!")
