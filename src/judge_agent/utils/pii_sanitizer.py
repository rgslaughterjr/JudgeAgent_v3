"""PII data generation and sanitization utilities"""

import re
import random
from typing import List, Tuple, Pattern
from dataclasses import dataclass


@dataclass
class PIIPattern:
    """PII pattern configuration"""
    name: str
    pattern: Pattern
    redaction_label: str
    description: str


class SyntheticDataGenerator:
    """
    Generate obviously fake synthetic PII data for testing.

    All generated data uses reserved/invalid ranges to ensure
    it's clearly synthetic and won't match real PII.

    Examples:
        >>> gen = SyntheticDataGenerator()
        >>> gen.ssn()
        '999-88-7654'
        >>> gen.credit_card()
        '4000-1234-5678-9010'
        >>> gen.phone()
        '(555) 123-4567'
    """

    def __init__(self, seed: int = None):
        """
        Initialize synthetic data generator.

        Args:
            seed: Random seed for reproducible generation
        """
        if seed is not None:
            random.seed(seed)

    def ssn(self) -> str:
        """
        Generate synthetic SSN starting with 999 (invalid area).

        Returns:
            SSN in format: 999-XX-XXXX
        """
        area = "999"  # Reserved, never issued
        group = f"{random.randint(0, 99):02d}"
        serial = f"{random.randint(0, 9999):04d}"
        return f"{area}-{group}-{serial}"

    def credit_card(self, card_type: str = "visa") -> str:
        """
        Generate synthetic credit card number using test ranges.

        Args:
            card_type: Type of card ('visa', 'mastercard', 'amex', 'discover')

        Returns:
            Credit card number in format: XXXX-XXXX-XXXX-XXXX
        """
        # Test card number prefixes
        prefixes = {
            'visa': '4000',
            'mastercard': '5000',
            'amex': '3700',
            'discover': '6011'
        }

        prefix = prefixes.get(card_type.lower(), '4000')

        if card_type.lower() == 'amex':
            # AmEx is 15 digits
            middle = f"{random.randint(0, 999999):06d}"
            last = f"{random.randint(0, 99999):05d}"
            return f"{prefix}-{middle}-{last}"
        else:
            # Visa, MC, Discover are 16 digits
            part2 = f"{random.randint(0, 9999):04d}"
            part3 = f"{random.randint(0, 9999):04d}"
            part4 = f"{random.randint(0, 9999):04d}"
            return f"{prefix}-{part2}-{part3}-{part4}"

    def phone(self, format_style: str = "formatted") -> str:
        """
        Generate synthetic phone number using 555 (reserved for fiction).

        Args:
            format_style: 'formatted' (555) 123-4567, 'dashes' 555-123-4567, 'plain' 5551234567

        Returns:
            Phone number in specified format
        """
        area = "555"  # Reserved for fictional use
        exchange = f"{random.randint(100, 999)}"
        line = f"{random.randint(0, 9999):04d}"

        if format_style == "formatted":
            return f"({area}) {exchange}-{line}"
        elif format_style == "dashes":
            return f"{area}-{exchange}-{line}"
        else:  # plain
            return f"{area}{exchange}{line}"

    def email(self, domain: str = "example.com") -> str:
        """
        Generate synthetic email address.

        Args:
            domain: Email domain (default: example.com, reserved by IANA)

        Returns:
            Email address like: test.user123@example.com
        """
        prefixes = ['test', 'demo', 'fake', 'sample', 'dummy']
        prefix = random.choice(prefixes)
        number = random.randint(100, 999)
        return f"{prefix}.user{number}@{domain}"

    def patient_id(self, prefix: str = "TEST") -> str:
        """
        Generate synthetic patient/medical ID.

        Args:
            prefix: Prefix to clearly mark as test data

        Returns:
            Patient ID like: TEST-P-123456789
        """
        number = f"{random.randint(100000000, 999999999)}"
        return f"{prefix}-P-{number}"

    def medical_record_number(self, facility: str = "DEMO") -> str:
        """
        Generate synthetic medical record number.

        Args:
            facility: Facility code prefix

        Returns:
            MRN like: DEMO-MRN-987654
        """
        number = f"{random.randint(100000, 999999)}"
        return f"{facility}-MRN-{number}"

    def date_of_birth(self) -> str:
        """
        Generate synthetic date of birth (obviously fake dates).

        Returns:
            DOB like: 01/01/1900
        """
        month = random.randint(1, 12)
        day = random.randint(1, 28)  # Safe for all months
        year = random.randint(1900, 1950)  # Obviously old
        return f"{month:02d}/{day:02d}/{year}"


class PIISanitizer:
    """
    Detect and redact PII from text strings.

    Supports detection of:
    - Social Security Numbers (SSN)
    - Credit card numbers
    - Phone numbers
    - Email addresses
    - Patient IDs
    - Medical record numbers

    Examples:
        >>> sanitizer = PIISanitizer()
        >>> text = "My SSN is 123-45-6789 and card is 4532-1234-5678-9010"
        >>> sanitizer.sanitize(text)
        'My SSN is [REDACTED-SSN] and card is [REDACTED-CC]'
    """

    def __init__(self, preserve_format: bool = False):
        """
        Initialize PII sanitizer.

        Args:
            preserve_format: If True, preserve length/format of redacted content
        """
        self.preserve_format = preserve_format
        self.patterns = self._compile_patterns()

    def _compile_patterns(self) -> List[PIIPattern]:
        """Compile regex patterns for PII detection."""

        patterns = [
            # More specific patterns first to avoid conflicts
            PIIPattern(
                name="patient_id",
                pattern=re.compile(
                    r'\b[A-Z]{2,6}[-]?P[-]?\d{6,9}\b'
                ),
                redaction_label="REDACTED-PATIENT-ID",
                description="Patient ID"
            ),
            PIIPattern(
                name="mrn",
                pattern=re.compile(
                    r'\b[A-Z]{2,6}[-]?MRN[-]?\d{5,8}\b'
                ),
                redaction_label="REDACTED-MRN",
                description="Medical Record Number"
            ),
            PIIPattern(
                name="ssn",
                pattern=re.compile(
                    r'\b\d{3}[-\s]\d{2}[-\s]\d{4}\b'  # Require separators
                ),
                redaction_label="REDACTED-SSN",
                description="Social Security Number"
            ),
            PIIPattern(
                name="credit_card",
                pattern=re.compile(
                    r'\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{4}[-\s]?\d{6}[-\s]?\d{5}\b'
                ),
                redaction_label="REDACTED-CC",
                description="Credit Card Number"
            ),
            PIIPattern(
                name="phone",
                pattern=re.compile(
                    r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
                ),
                redaction_label="REDACTED-PHONE",
                description="Phone Number"
            ),
            PIIPattern(
                name="email",
                pattern=re.compile(
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                ),
                redaction_label="REDACTED-EMAIL",
                description="Email Address"
            ),
            PIIPattern(
                name="date_of_birth",
                pattern=re.compile(
                    r'\b(?:DOB|Date of Birth):\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                    re.IGNORECASE
                ),
                redaction_label="REDACTED-DOB",
                description="Date of Birth"
            ),
        ]

        return patterns

    def sanitize(self, text: str) -> str:
        """
        Sanitize PII from text by replacing with redaction labels.

        Args:
            text: Text potentially containing PII

        Returns:
            Sanitized text with PII replaced by [REDACTED-*] markers

        Examples:
            >>> sanitizer = PIISanitizer()
            >>> sanitizer.sanitize("Call me at 555-123-4567")
            'Call me at [REDACTED-PHONE]'
        """
        if not text:
            return text

        sanitized = text

        # Apply each pattern
        for pii_pattern in self.patterns:
            if self.preserve_format:
                # Replace with X's preserving original length
                def replace_with_x(match):
                    return 'X' * len(match.group(0))
                sanitized = pii_pattern.pattern.sub(replace_with_x, sanitized)
            else:
                # Replace with redaction label
                sanitized = pii_pattern.pattern.sub(
                    f"[{pii_pattern.redaction_label}]",
                    sanitized
                )

        return sanitized

    def detect_pii(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        Detect PII in text without sanitizing.

        Args:
            text: Text to scan for PII

        Returns:
            List of tuples: (pii_type, matched_text, start_pos, end_pos)

        Examples:
            >>> sanitizer = PIISanitizer()
            >>> sanitizer.detect_pii("Email: test@example.com")
            [('email', 'test@example.com', 7, 24)]
        """
        if not text:
            return []

        detections = []

        for pii_pattern in self.patterns:
            for match in pii_pattern.pattern.finditer(text):
                detections.append((
                    pii_pattern.name,
                    match.group(0),
                    match.start(),
                    match.end()
                ))

        # Sort by position
        detections.sort(key=lambda x: x[2])

        return detections

    def has_pii(self, text: str) -> bool:
        """
        Check if text contains any PII.

        Args:
            text: Text to check

        Returns:
            True if PII detected, False otherwise
        """
        return len(self.detect_pii(text)) > 0

    def get_pii_count(self, text: str) -> dict:
        """
        Count PII occurrences by type.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with counts: {'ssn': 2, 'email': 1, ...}
        """
        detections = self.detect_pii(text)
        counts = {}

        for pii_type, _, _, _ in detections:
            counts[pii_type] = counts.get(pii_type, 0) + 1

        return counts


# Convenience functions
def sanitize_text(text: str, preserve_format: bool = False) -> str:
    """
    Convenience function to sanitize text.

    Args:
        text: Text to sanitize
        preserve_format: Preserve original format with X's

    Returns:
        Sanitized text
    """
    sanitizer = PIISanitizer(preserve_format=preserve_format)
    return sanitizer.sanitize(text)


def generate_test_data() -> dict:
    """
    Generate a set of synthetic test data.

    Returns:
        Dictionary containing various synthetic PII fields
    """
    gen = SyntheticDataGenerator()

    return {
        'ssn': gen.ssn(),
        'credit_card_visa': gen.credit_card('visa'),
        'credit_card_mastercard': gen.credit_card('mastercard'),
        'credit_card_amex': gen.credit_card('amex'),
        'phone_formatted': gen.phone('formatted'),
        'phone_dashes': gen.phone('dashes'),
        'phone_plain': gen.phone('plain'),
        'email': gen.email(),
        'email_test': gen.email('test.example.org'),
        'patient_id': gen.patient_id(),
        'mrn': gen.medical_record_number(),
        'dob': gen.date_of_birth()
    }
