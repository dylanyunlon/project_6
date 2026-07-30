#!/usr/bin/env python3
"""muh/parse.py — Parse .muh configuration files

A .muh file is YAML with these semantics:
  - 'extends': inherit from another .muh file (deep merge, child overrides parent)
  - 'hardware': target hardware description (warp_size, smem, registers, etc.)
  - 'tuning': per-algorithm parameter overrides
  - 'vllm': vllm-specific launch config (maps to computility-run.yaml)
  - 'env': environment variable overrides

Schema validation against muh/schema/*.yaml ensures parameter names and
value ranges are legal.

Usage:
    python3 muh/parse.py baseline.muh [--schema-dir muh/schema] [--validate]
"""

import os
import sys
import copy
import argparse
from pathlib import Path


def yaml_load_simple(text):
    """Minimal YAML parser — handles flat dicts, nested dicts, lists, strings, numbers.
    No dependency on PyYAML. Sufficient for .muh files."""
    result = {}
    stack = [(result, -1)]  # (dict, indent_level)
    current_list_key = None

    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue

        indent = len(line) - len(line.lstrip())

        # Pop stack to find parent at correct indent
        while len(stack) > 1 and stack[-1][1] >= indent:
            stack.pop()

        parent, _ = stack[-1]

        # List item
        if stripped.startswith('- '):
            val = stripped[2:].strip()
            if current_list_key and current_list_key in parent:
                if not isinstance(parent[current_list_key], list):
                    parent[current_list_key] = []
                parent[current_list_key].append(_parse_value(val))
            continue

        if ':' in stripped:
            key, _, val = stripped.partition(':')
            key = key.strip()
            val = val.strip()

            if val == '' or val == '|':
                # Nested dict or upcoming list
                parent[key] = {}
                stack.append((parent[key], indent))
                current_list_key = key
            elif val.startswith('[') and val.endswith(']'):
                # Inline list
                items = [_parse_value(v.strip()) for v in val[1:-1].split(',') if v.strip()]
                parent[key] = items
                current_list_key = None
            else:
                parent[key] = _parse_value(val)
                current_list_key = key if val == '' else None

    return result


def _parse_value(v):
    """Parse a YAML scalar value."""
    if v in ('true', 'True', 'yes'):
        return True
    if v in ('false', 'False', 'no'):
        return False
    if v in ('null', 'None', '~', 'TBD'):
        return None
    # Strip quotes
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    # Try int
    try:
        return int(v)
    except ValueError:
        pass
    # Try float
    try:
        return float(v)
    except ValueError:
        pass
    return v


def deep_merge(base, override):
    """Deep merge two dicts; override wins on conflicts."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_muh(filepath, search_dirs=None):
    """Load a .muh file, resolving 'extends' chain."""
    if search_dirs is None:
        search_dirs = [os.path.dirname(filepath), '.']

    with open(filepath, 'r') as f:
        data = yaml_load_simple(f.read())

    # Resolve extends
    if 'extends' in data:
        parent_name = data.pop('extends')
        parent_path = None
        for d in search_dirs:
            candidate = os.path.join(d, parent_name)
            if os.path.exists(candidate):
                parent_path = candidate
                break
        if parent_path is None:
            raise FileNotFoundError(f"Cannot find parent .muh file: {parent_name} (searched {search_dirs})")
        parent_data = load_muh(parent_path, search_dirs)
        data = deep_merge(parent_data, data)

    return data


def load_schema(schema_dir, algo):
    """Load a schema YAML for validation."""
    path = os.path.join(schema_dir, f"{algo}.yaml")
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return yaml_load_simple(f.read())


def validate_tuning(config, schema_dir):
    """Validate tuning parameters against extracted schemas."""
    errors = []
    tuning = config.get('tuning', {})

    for algo, params in tuning.items():
        if not isinstance(params, dict):
            continue
        schema = load_schema(schema_dir, algo)
        if schema is None:
            errors.append(f"WARNING: No schema for algorithm '{algo}' — skipping validation")
            continue

        schema_params = schema.get('parameters', {})
        for param_name, param_value in params.items():
            if param_name.startswith('_'):  # metadata keys
                continue
            if param_name not in schema_params:
                errors.append(f"{algo}.{param_name}: unknown parameter (not in schema)")
                continue

            spec = schema_params[param_name]
            if spec.get('type') == 'int' and isinstance(param_value, (int, float)):
                rng = spec.get('range', [0, 99999])
                if not (rng[0] <= param_value <= rng[1]):
                    errors.append(
                        f"{algo}.{param_name}: value {param_value} outside range {rng}"
                    )
            elif spec.get('type') == 'enum' and isinstance(param_value, str):
                valid = spec.get('values', [])
                if param_value not in valid:
                    errors.append(
                        f"{algo}.{param_name}: '{param_value}' not in {valid}"
                    )

    return errors


def print_config(config, indent=0):
    """Pretty-print a parsed .muh config."""
    prefix = "  " * indent
    for k, v in config.items():
        if isinstance(v, dict):
            print(f"{prefix}{k}:")
            print_config(v, indent + 1)
        elif isinstance(v, list):
            print(f"{prefix}{k}:")
            for item in v:
                print(f"{prefix}  - {item}")
        else:
            print(f"{prefix}{k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Parse and validate .muh files")
    parser.add_argument("muh_file", help="Path to .muh file")
    parser.add_argument("--schema-dir", default="muh/schema",
                        help="Schema directory (default: muh/schema)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate tuning params against schemas")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of pretty-print")
    args = parser.parse_args()

    config = load_muh(args.muh_file)

    if args.validate:
        errors = validate_tuning(config, args.schema_dir)
        if errors:
            print("Validation errors:", file=sys.stderr)
            for e in errors:
                print(f"  ✗ {e}", file=sys.stderr)
            sys.exit(1)
        else:
            print("Validation passed ✓", file=sys.stderr)

    if args.json:
        import json
        print(json.dumps(config, indent=2, ensure_ascii=False))
    else:
        print_config(config)


if __name__ == "__main__":
    main()
