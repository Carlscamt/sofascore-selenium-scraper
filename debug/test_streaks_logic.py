import re

def slugify(text):
    """Converts 'More than 2.5 goals' to 'more_than_2_5_goals'."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')

def parse_streak_value(val_str):
    """
    Parses streak value string (e.g. '3' or '6/7').
    Returns dict with count, sample, pct.
    """
    try:
        if '/' in str(val_str):
            parts = str(val_str).split('/')
            count = int(parts[0])
            sample = int(parts[1])
            pct = round(count / sample, 3) if sample > 0 else 0.0
        else:
            # Just a number like "3"
            count = int(val_str)
            sample = count # Implied sample is the streak itself
            pct = 1.0
            
        return {
            'count': count,
            'sample': sample,
            'pct': pct
        }
    except Exception as e:
        print(f"Error parsing '{val_str}': {e}")
        return None

def test_logic():
    test_cases = [
        ("3", {'count': 3, 'sample': 3, 'pct': 1.0}),
        ("6/7", {'count': 6, 'sample': 7, 'pct': 0.857}),
        ("5/5", {'count': 5, 'sample': 5, 'pct': 1.0}),
        ("0", {'count': 0, 'sample': 0, 'pct': 1.0}), # Edge case
    ]
    
    print("Testing Value Parsing...")
    for input_val, expected in test_cases:
        result = parse_streak_value(input_val)
        print(f"Input: {input_val} -> Result: {result}")
        assert result == expected, f"Failed for {input_val}"
        
    print("\nTesting Slugify...")
    slug_tests = [
        ("More than 2.5 goals", "more_than_2_5_goals"),
        ("No wins", "no_wins"),
        ("First to concede", "first_to_concede"),
        ("Without clean sheet", "without_clean_sheet"),
    ]
    for text, expected in slug_tests:
        res = slugify(text)
        print(f"Input: '{text}' -> Slug: '{res}'")
        assert res == expected, f"Failed for {text}"

    print("\nALL TESTS PASSED")

if __name__ == "__main__":
    test_logic()
