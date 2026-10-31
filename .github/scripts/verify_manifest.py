#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
import jsonschema
from packaging.version import Version

ASSET_SCHEMA = {
    "type": "object",
    "required": ["url", "sha256", "size"],
    "additionalProperties": False,
    "properties": {
        "url": {"type": "string", "pattern": r"^https://"},
        "sha256": {"type": "string", "pattern": r"^[0-9a-fA-F]{64}$"},
        "size": {"type": "integer", "minimum": 1},
    },
}

SAMPLING_SCHEMA = {
    "type": "object",
    "required": [
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "chat_template_kwargs",
    ],
    "additionalProperties": False,
    "properties": {
        "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
        "top_p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "top_k": {"type": "integer", "minimum": 0},
        "min_p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "presence_penalty": {"type": "number", "minimum": -2.0, "maximum": 2.0},
        "reasoning_effort": {"type": "string", "enum": ["low", "medium", "high", "xhigh"]},
        "chat_template_kwargs": {
            "type": "object",
            "required": ["enable_thinking"],
            "additionalProperties": False,
            "properties": {
                "enable_thinking": {"type": "boolean"},
                "preserve_thinking": {"type": "boolean"},
                "reasoning_effort": {"type": "string", "enum": ["low", "medium", "high", "xhigh"]},
            },
        },
    },
}

TIER_SCHEMA = {
    "type": "object",
    "required": [
        "version",
        "model",
        "size_bytes",
        "sha256",
        "min_llama_cpp",
        "server_args",
        "request_defaults",
        "license",
        "urls",
        "vision_to_run_on",
        "vision_model",
        "vision_model_size_bytes",
        "vision_model_sha256",
        "vision_urls",
    ],
    "additionalProperties": False,
    "properties": {
        "version": {"type": "string", "pattern": r"^\d+(\.\d+)*$"},
        "model": {"type": "string", "minLength": 1},
        "size_bytes": {"type": "integer", "minimum": 1},
        "sha256": {"type": "string", "pattern": r"^[0-9a-fA-F]{64}$"},
        "min_llama_cpp": {"type": "string", "pattern": r"^b\d+$"},
        "server_args": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
        },
        "request_defaults": {
            "type": "object",
            "required": ["normal", "code_generation"],
            "additionalProperties": False,
            "properties": {
                "normal": SAMPLING_SCHEMA,
                "code_generation": SAMPLING_SCHEMA,
            },
        },
        "license": {"type": "string", "minLength": 1},
        "urls": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "pattern": r"^https://"},
        },
        "vision_to_run_on": {"type": "string", "enum": ["cpu", "gpu"]},
        "vision_model": {"type": "string", "minLength": 1},
        "vision_model_size_bytes": {"type": "integer", "minimum": 1},
        "vision_model_sha256": {"type": "string", "pattern": r"^[0-9a-fA-F]{64}$"},
        "vision_urls": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "pattern": r"^https://"},
        },
    },
}

MANIFEST_SCHEMA = {
    "type": "object",
    "required": ["version", "llama.cpp", "tiers"],
    "additionalProperties": False,
    "properties": {
        "version": {"type": "integer", "minimum": 1},
        "llama.cpp": {
            "type": "object",
            "required": ["version", "assets"],
            "additionalProperties": False,
            "properties": {
                "version": {"type": "string", "pattern": r"^b\d+$"},
                "assets": {
                    "type": "object",
                    "required": ["cuda13", "cuda12", "vulkan", "cpu"],
                    "additionalProperties": False,
                    "properties": {
                        "cuda13": {
                            "type": "object",
                            "required": ["main", "cudart"],
                            "additionalProperties": False,
                            "properties": {"main": ASSET_SCHEMA, "cudart": ASSET_SCHEMA},
                        },
                        "cuda12": {
                            "type": "object",
                            "required": ["main", "cudart"],
                            "additionalProperties": False,
                            "properties": {"main": ASSET_SCHEMA, "cudart": ASSET_SCHEMA},
                        },
                        "vulkan": ASSET_SCHEMA,
                        "cpu": ASSET_SCHEMA,
                    },
                },
            },
        },
        "tiers": {
            "type": "object",
            "minProperties": 1,
            "patternProperties": {
                r"^vram_\d+gb$": TIER_SCHEMA,
            },
            "additionalProperties": False,
        },
    },
}


def parse_llama_version(v_str: str) -> int:
    match = re.fullmatch(r"b(\d+)", str(v_str).strip())
    if not match:
        raise ValueError(f"Invalid llama.cpp version '{v_str}' (expected format 'b11946')")
    return int(match.group(1))


def verify_schema(manifest: dict) -> list[str]:
    """Validates JSON structure, types, and cross-field consistency."""
    errors = []
    validator = jsonschema.Draft202012Validator(MANIFEST_SCHEMA)
    for err in sorted(validator.iter_errors(manifest), key=lambda e: list(e.path)):
        path = ".".join(str(p) for p in err.path) or "<root>"
        errors.append(f"Schema error at '{path}': {err.message}")

    if errors:
        return errors

    # Cross-field check: tier min_llama_cpp <= llama.cpp.version
    top_llama_ver = manifest["llama.cpp"]["version"]
    top_llama_num = parse_llama_version(top_llama_ver)

    for tier_name, tier in manifest["tiers"].items():
        min_num = parse_llama_version(tier["min_llama_cpp"])
        if min_num > top_llama_num:
            errors.append(
                f"Tier '{tier_name}' requires min_llama_cpp {tier['min_llama_cpp']}, "
                f"which is newer than manifest llama.cpp.version ({top_llama_ver})"
            )

    return errors


def check_url_reachable(url: str, headers: dict) -> str | None:
    req = urllib.request.Request(url, headers=headers, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                return f"Returned HTTP {resp.status} instead of 200: {url}"
    except urllib.error.HTTPError as e:
        return f"Broken link ({e.code} {e.reason}): {url}"
    except Exception as e:
        return f"Unreachable link ({e}): {url}"
    return None


def verify_huggingface_url(url: str, expected_size: int, expected_sha256: str) -> list[str]:
    errors = []
    headers = {"User-Agent": "manifest-verifier/1.0"}

    if "/resolve/" not in url:
        return [f"Invalid Hugging Face download URL (missing '/resolve/'): {url}"]

    reach_err = check_url_reachable(url, headers)
    if reach_err:
        return [reach_err]

    raw_url = url.replace("/resolve/", "/raw/", 1)
    req = urllib.request.Request(raw_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            lfs_text = resp.read(2048).decode("utf-8", errors="replace")
    except Exception as e:
        return [f"Failed to fetch LFS pointer from {raw_url}: {e}"]

    sha_match = re.search(r"^oid sha256:([0-9a-fA-F]{64})$", lfs_text, re.MULTILINE)
    size_match = re.search(r"^size (\d+)$", lfs_text, re.MULTILINE)

    if not sha_match or not size_match:
        return [f"URL did not return a valid Git LFS pointer at {raw_url}"]

    actual_sha256 = sha_match.group(1).lower()
    actual_size = int(size_match.group(1))

    if actual_size != expected_size:
        errors.append(f"Size mismatch for {url}: manifest has {expected_size}, LFS has {actual_size}")
    if actual_sha256 != expected_sha256.lower():
        errors.append(f"SHA256 mismatch for {url}: manifest has {expected_sha256}, LFS has {actual_sha256}")

    return errors


def verify_github_url(url: str, expected_size: int, expected_sha256: str) -> list[str]:
    errors = []
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "manifest-verifier/1.0"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    reach_err = check_url_reachable(url, {"User-Agent": "manifest-verifier/1.0"})
    if reach_err:
        return [reach_err]

    match = re.match(
        r"^https://github\.com/([^/]+)/([^/]+)/releases/download/([^/]+)/([^/]+)$", url
    )
    if not match:
        return [f"Unrecognized GitHub release URL format: {url}"]

    owner, repo, tag, filename = match.groups()
    api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
    req = urllib.request.Request(api_url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            release_data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return [f"Failed to query GitHub API {api_url}: {e}"]

    asset_info = next((a for a in release_data.get("assets", []) if a.get("name") == filename), None)
    if not asset_info:
        return [f"Asset '{filename}' not found in GitHub release {tag}"]

    actual_size = asset_info.get("size")
    if actual_size != expected_size:
        errors.append(f"Size mismatch for {filename}: manifest has {expected_size}, GitHub has {actual_size}")

    digest = asset_info.get("digest") or ""
    if digest.startswith("sha256:"):
        actual_sha256 = digest.removeprefix("sha256:").lower()
        if actual_sha256 != expected_sha256.lower():
            errors.append(f"SHA256 mismatch for {filename}: manifest has {expected_sha256}, GitHub has {actual_sha256}")

    return errors


def verify_versions(current: dict, base: dict) -> list[str]:
    errors = []
    curr_root = Version(str(current.get("version", "0")))
    base_root = Version(str(base.get("version", "0")))
    if curr_root < base_root:
        errors.append(f"Root 'version' decreased from {base.get('version')} to {current.get('version')}")
    elif current != base and curr_root <= base_root:
        errors.append(f"Manifest changed, but root 'version' ({current.get('version')}) was not incremented")

    curr_llama = current.get("llama.cpp", {})
    base_llama = base.get("llama.cpp", {})
    if curr_llama and base_llama:
        curr_b = parse_llama_version(curr_llama.get("version", ""))
        base_b = parse_llama_version(base_llama.get("version", ""))
        if curr_b < base_b:
            errors.append(f"llama.cpp version decreased from b{base_b} to b{curr_b}")
        elif curr_llama != base_llama and curr_b <= base_b:
            errors.append(f"llama.cpp section changed, but llama.cpp.version (b{curr_b}) was not incremented")

    curr_tiers = current.get("tiers", {})
    base_tiers = base.get("tiers", {})
    for tier_name, curr_tier in curr_tiers.items():
        if tier_name in base_tiers:
            base_tier = base_tiers[tier_name]
            curr_tv = Version(str(curr_tier.get("version", "0")))
            base_tv = Version(str(base_tier.get("version", "0")))
            if curr_tv < base_tv:
                errors.append(f"Tier '{tier_name}' version decreased from {base_tv} to {curr_tv}")
            elif curr_tier != base_tier and curr_tv <= base_tv:
                errors.append(f"Tier '{tier_name}' changed, but its version ({curr_tv}) was not incremented")

    return errors


def collect_artifacts(manifest: dict) -> list[tuple[str, str, int, str]]:
    items = []
    assets = manifest.get("llama.cpp", {}).get("assets", {})
    for backend, val in assets.items():
        if "url" in val:
            items.append((f"llama.cpp.{backend}", val["url"], val["size"], val["sha256"]))
        else:
            for sub_k, sub_v in val.items():
                if isinstance(sub_v, dict) and "url" in sub_v:
                    items.append((f"llama.cpp.{backend}.{sub_k}", sub_v["url"], sub_v["size"], sub_v["sha256"]))

    for tier_name, tier in manifest.get("tiers", {}).items():
        for url in tier.get("urls", []):
            items.append((f"{tier_name}.model", url, tier["size_bytes"], tier["sha256"]))
        for v_url in tier.get("vision_urls", []):
            items.append((f"{tier_name}.vision", v_url, tier["vision_model_size_bytes"], tier["vision_model_sha256"]))

    return items


def main():
    parser = argparse.ArgumentParser(description="Verify offline_manifest.json")
    parser.add_argument("manifest", nargs="?", default="config/offline_manifest.json")
    parser.add_argument("--base", help="Path to previous manifest JSON for version comparison")
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        current = json.load(f)

    # 1. Validate Schema First
    print("Validating JSON Schema...")
    schema_errors = verify_schema(current)
    if schema_errors:
        print("\n=== SCHEMA VALIDATION FAILED ===")
        for err in schema_errors:
            print(f" - {err}")
        sys.exit(1)

    all_errors = []

    # 2. Validate Version Increments
    if args.base and os.path.exists(args.base):
        with open(args.base, "r", encoding="utf-8") as f:
            base = json.load(f)
        all_errors.extend(verify_versions(current, base))

    # 3. Validate URLs, Sizes, and Hashes
    for name, url, size, sha256 in collect_artifacts(current):
        print(f"Checking [{name}] -> {url}")
        if "huggingface.co" in url:
            errs = verify_huggingface_url(url, size, sha256)
        elif "github.com" in url:
            errs = verify_github_url(url, size, sha256)
        else:
            errs = [f"Unsupported host for URL: {url}"]

        for e in errs:
            print(f"  ❌ {e}")
        all_errors.extend([f"[{name}] {e}" for e in errs])

    if all_errors:
        print("\n=== MANIFEST VERIFICATION FAILED ===")
        for err in all_errors:
            print(f" - {err}")
        sys.exit(1)

    print("\n✅ Schema, URLs, sizes, SHA-256 hashes, and versions verified!")


if __name__ == "__main__":
    main()
