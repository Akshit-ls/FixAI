#!/usr/bin/env python3
import argparse
import base64
import os
import sys
from cryptography.hazmat.primitives.asymmetric import ed25519

def parse_args():
    parser = argparse.ArgumentParser(description="Sign a file using Ed25519 private key.")
    # Supports both --input / --output flags and positional arguments
    parser.add_argument("pos_input", nargs="?", help="Input file path")
    parser.add_argument("pos_output", nargs="?", help="Output signature file path")
    parser.add_argument("--input", "-i", dest="flag_input", help="Input file path")
    parser.add_argument("--output", "-o", dest="flag_output", help="Output signature file path")
    
    args = parser.parse_args()
    input_file = args.flag_input or args.pos_input or "config/offline_manifest.json"
    output_file = args.flag_output or args.pos_output or f"{input_file}.sig"
    return input_file, output_file

def main():
    input_file, output_file = parse_args()

    # Read base64 private key from environment
    # Check both possible env var names you've used
    private_key_b64 = os.getenv("PRIVATE_KEY_B64") or os.getenv("SIGNING_PRIVATE_KEY")
    if not private_key_b64:
        print("Error: Missing private key environment variable (PRIVATE_KEY_B64 or SIGNING_PRIVATE_KEY).")
        sys.exit(1)

    try:
        # Decode base64 to 32 raw bytes
        private_bytes = base64.b64decode(private_key_b64.strip())
        private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_bytes)
    except Exception as e:
        print(f"Error loading private key: {e}")
        sys.exit(1)

    # Read target file bytes
    if not os.path.exists(input_file):
        print(f"Error: Target file '{input_file}' not found.")
        sys.exit(1)

    with open(input_file, "rb") as f:
        data = f.read()

    # Sign the bytes
    signature = private_key.sign(data)

    # Ensure target directory exists and save signature
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "wb") as f:
        f.write(signature)

    print(f"Successfully signed: {input_file} -> {output_file}")

if __name__ == "__main__":
    main()
