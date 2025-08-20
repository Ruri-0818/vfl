#!/usr/bin/env python3
"""
Clean all non-ASCII characters from the Python file
"""

import re

def clean_file():
    print("Reading file...")
    
    # Read file with error handling
    content = ""
    try:
        with open('train_bank_villain_with_inference.py', 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        try:
            with open('train_bank_villain_with_inference.py', 'r', encoding='gbk', errors='ignore') as f:
                content = f.read()
        except:
            with open('train_bank_villain_with_inference.py', 'rb') as f:
                content = f.read().decode('utf-8', errors='ignore')
    
    print("Cleaning non-ASCII characters...")
    
    # Remove all non-ASCII characters and replace with space
    cleaned_content = ""
    for char in content:
        if ord(char) < 128:  # ASCII characters only
            cleaned_content += char
        elif char == '\n':
            cleaned_content += char
        elif char == '\t':
            cleaned_content += char
        else:
            cleaned_content += ' '  # Replace non-ASCII with space
    
    # Clean up multiple spaces
    cleaned_content = re.sub(r' +', ' ', cleaned_content)
    
    # Fix common issues
    cleaned_content = cleaned_content.replace('( ', '(')
    cleaned_content = cleaned_content.replace(' )', ')')
    cleaned_content = cleaned_content.replace(' ,', ',')
    cleaned_content = cleaned_content.replace(' .', '.')
    cleaned_content = cleaned_content.replace(' :', ':')
    cleaned_content = cleaned_content.replace(' =', '=')
    cleaned_content = cleaned_content.replace('= ', '=')
    
    # Fix print statements that may have been corrupted
    print_fixes = {
        'print(f" Creating Bank Marketing model': 'print(f"Creating Bank Marketing model',
        'print(f" Created malicious bottom model': 'print(f"Created malicious bottom model',
        'print(f" Created normal bottom model': 'print(f"Created normal bottom model',
        'print(f" Created top model': 'print(f"Created top model',
        'print(f" VILLAIN trigger set to party': 'print(f"VILLAIN trigger set to party',
        'print(f" Model creation completed': 'print(f"Model creation completed',
        'print(f" ': 'print(f"',
        'print(" ': 'print("',
    }
    
    for broken, fixed in print_fixes.items():
        cleaned_content = cleaned_content.replace(broken, fixed)
    
    print("Writing cleaned file...")
    
    # Write cleaned content
    with open('train_bank_villain_with_inference.py', 'w', encoding='ascii', errors='ignore') as f:
        f.write(cleaned_content)
    
    print("File cleaned successfully!")
    
    # Verify the file can be read properly
    try:
        with open('train_bank_villain_with_inference.py', 'r', encoding='utf-8') as f:
            test_content = f.read()
        print("Verification: File can be read with UTF-8 encoding.")
    except Exception as e:
        print(f"Warning: {e}")

if __name__ == '__main__':
    clean_file() 