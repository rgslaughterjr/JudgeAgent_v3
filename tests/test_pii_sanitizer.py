"""Test PII sanitizer and synthetic data generator"""

from utils.pii_sanitizer import (
    SyntheticDataGenerator,
    PIISanitizer,
    sanitize_text,
    generate_test_data
)


def test_synthetic_data_generator():
    """Test synthetic data generation"""
    print("=" * 60)
    print("Testing SyntheticDataGenerator")
    print("=" * 60)

    gen = SyntheticDataGenerator(seed=42)

    # Test SSN generation
    ssn = gen.ssn()
    assert ssn.startswith("999-"), f"SSN should start with 999-, got {ssn}"
    print(f"[PASS] SSN generation: {ssn}")

    # Test credit card generation
    visa = gen.credit_card('visa')
    assert visa.startswith("4000"), f"Visa should start with 4000, got {visa}"
    print(f"[PASS] Visa generation: {visa}")

    mastercard = gen.credit_card('mastercard')
    assert mastercard.startswith("5000"), f"MC should start with 5000, got {mastercard}"
    print(f"[PASS] Mastercard generation: {mastercard}")

    amex = gen.credit_card('amex')
    assert amex.startswith("3700"), f"AmEx should start with 3700, got {amex}"
    assert len(amex.replace('-', '')) == 15, "AmEx should be 15 digits"
    print(f"[PASS] AmEx generation: {amex}")

    # Test phone generation
    phone_formatted = gen.phone('formatted')
    assert phone_formatted.startswith("(555)"), f"Phone should start with (555), got {phone_formatted}"
    print(f"[PASS] Phone (formatted): {phone_formatted}")

    phone_dashes = gen.phone('dashes')
    assert phone_dashes.startswith("555-"), f"Phone should start with 555-, got {phone_dashes}"
    print(f"[PASS] Phone (dashes): {phone_dashes}")

    phone_plain = gen.phone('plain')
    assert phone_plain.startswith("555"), f"Phone should start with 555, got {phone_plain}"
    print(f"[PASS] Phone (plain): {phone_plain}")

    # Test email generation
    email = gen.email()
    assert "@example.com" in email, f"Email should contain @example.com, got {email}"
    assert email.split('@')[0].startswith(('test', 'demo', 'fake', 'sample', 'dummy')), \
        f"Email should have test prefix, got {email}"
    print(f"[PASS] Email generation: {email}")

    # Test patient ID
    patient_id = gen.patient_id()
    assert patient_id.startswith("TEST-P-"), f"Patient ID should start with TEST-P-, got {patient_id}"
    print(f"[PASS] Patient ID: {patient_id}")

    # Test MRN
    mrn = gen.medical_record_number()
    assert "MRN" in mrn, f"MRN should contain 'MRN', got {mrn}"
    print(f"[PASS] Medical Record Number: {mrn}")

    # Test DOB
    dob = gen.date_of_birth()
    assert len(dob) == 10, f"DOB should be MM/DD/YYYY format, got {dob}"
    print(f"[PASS] Date of Birth: {dob}")

    print()


def test_pii_sanitizer():
    """Test PII detection and sanitization"""
    print("=" * 60)
    print("Testing PIISanitizer")
    print("=" * 60)

    sanitizer = PIISanitizer()

    # Test SSN sanitization
    text = "My SSN is 123-45-6789 and I was born in 1990."
    sanitized = sanitizer.sanitize(text)
    assert "[REDACTED-SSN]" in sanitized, "Should redact SSN"
    assert "123-45-6789" not in sanitized, "Should not contain original SSN"
    print(f"[PASS] SSN sanitization")
    print(f"  Original: {text}")
    print(f"  Sanitized: {sanitized}")

    # Test credit card sanitization
    text = "Please charge card 4532-1234-5678-9010 for the purchase."
    sanitized = sanitizer.sanitize(text)
    assert "[REDACTED-CC]" in sanitized, "Should redact credit card"
    assert "4532" not in sanitized, "Should not contain card number"
    print(f"[PASS] Credit card sanitization")
    print(f"  Original: {text}")
    print(f"  Sanitized: {sanitized}")

    # Test phone sanitization
    text = "Call me at (555) 123-4567 or 555-987-6543."
    sanitized = sanitizer.sanitize(text)
    assert sanitized.count("[REDACTED-PHONE]") == 2, "Should redact both phone numbers"
    print(f"[PASS] Phone sanitization")
    print(f"  Original: {text}")
    print(f"  Sanitized: {sanitized}")

    # Test email sanitization
    text = "Contact john.doe@example.com for more info."
    sanitized = sanitizer.sanitize(text)
    assert "[REDACTED-EMAIL]" in sanitized, "Should redact email"
    assert "@example.com" not in sanitized, "Should not contain email"
    print(f"[PASS] Email sanitization")
    print(f"  Original: {text}")
    print(f"  Sanitized: {sanitized}")

    # Test multiple PII types
    text = "Patient 123-45-6789 at john@example.com, phone (555) 123-4567, card 4532123456789010"
    sanitized = sanitizer.sanitize(text)
    assert "[REDACTED-SSN]" in sanitized
    assert "[REDACTED-EMAIL]" in sanitized
    assert "[REDACTED-PHONE]" in sanitized
    assert "[REDACTED-CC]" in sanitized
    print(f"[PASS] Multiple PII types")
    print(f"  Original: {text}")
    print(f"  Sanitized: {sanitized}")

    # Test patient ID sanitization
    text = "Patient ID: TEST-P-123456789 requires attention."
    sanitized = sanitizer.sanitize(text)
    assert "[REDACTED-PATIENT-ID]" in sanitized, "Should redact patient ID"
    print(f"[PASS] Patient ID sanitization")
    print(f"  Original: {text}")
    print(f"  Sanitized: {sanitized}")

    print()


def test_pii_detection():
    """Test PII detection without sanitization"""
    print("=" * 60)
    print("Testing PII Detection")
    print("=" * 60)

    sanitizer = PIISanitizer()

    text = "SSN: 123-45-6789, Email: test@example.com, Phone: 555-123-4567"
    detections = sanitizer.detect_pii(text)

    print(f"Text: {text}")
    print(f"Detections found: {len(detections)}")

    for pii_type, matched_text, start, end in detections:
        print(f"  - {pii_type}: '{matched_text}' at position {start}-{end}")

    assert len(detections) == 3, f"Should detect 3 PII items, found {len(detections)}"
    print(f"[PASS] PII detection")

    # Test has_pii
    assert sanitizer.has_pii(text), "Should detect PII presence"
    assert not sanitizer.has_pii("No PII here"), "Should not detect PII in clean text"
    print(f"[PASS] has_pii() method")

    # Test get_pii_count
    counts = sanitizer.get_pii_count(text)
    assert counts.get('ssn', 0) == 1, "Should count 1 SSN"
    assert counts.get('email', 0) == 1, "Should count 1 email"
    print(f"[PASS] get_pii_count() method: {counts}")

    print()


def test_preserve_format():
    """Test format-preserving sanitization"""
    print("=" * 60)
    print("Testing Format Preservation")
    print("=" * 60)

    sanitizer = PIISanitizer(preserve_format=True)

    text = "SSN: 123-45-6789"
    sanitized = sanitizer.sanitize(text)

    print(f"Original:  {text}")
    print(f"Sanitized: {sanitized}")

    # Check that length is preserved
    original_numbers = text.replace("SSN: ", "")
    sanitized_numbers = sanitized.replace("SSN: ", "")
    assert len(original_numbers) == len(sanitized_numbers), "Should preserve length"
    assert "X" in sanitized, "Should use X for redaction"
    print(f"[PASS] Format preservation")

    print()


def test_convenience_functions():
    """Test convenience functions"""
    print("=" * 60)
    print("Testing Convenience Functions")
    print("=" * 60)

    # Test sanitize_text
    result = sanitize_text("My SSN is 123-45-6789")
    assert "[REDACTED-SSN]" in result, "sanitize_text should work"
    print(f"[PASS] sanitize_text() function")

    # Test generate_test_data
    test_data = generate_test_data()
    assert 'ssn' in test_data, "Should generate SSN"
    assert 'email' in test_data, "Should generate email"
    assert 'phone_formatted' in test_data, "Should generate phone"
    assert test_data['ssn'].startswith('999-'), "SSN should be obviously fake"
    print(f"[PASS] generate_test_data() function")
    print("Sample test data:")
    for key, value in list(test_data.items())[:5]:
        print(f"  {key}: {value}")

    print()


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("PII Sanitizer Test Suite")
    print("=" * 60 + "\n")

    test_synthetic_data_generator()
    test_pii_sanitizer()
    test_pii_detection()
    test_preserve_format()
    test_convenience_functions()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
