"""Query OSV for every locked registry dependency; fail closed on API errors."""
import json
import pathlib
import tomllib
import urllib.request


def main():
    lock = tomllib.loads(pathlib.Path("Cargo.lock").read_text())
    packages = [p for p in lock["package"] if "source" in p]
    queries = [
        {"package": {"name": p["name"], "ecosystem": "crates.io"}, "version": p["version"]}
        for p in packages
    ]
    req = urllib.request.Request(
        "https://api.osv.dev/v1/querybatch",
        data=json.dumps({"queries": queries}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        results = json.load(response)["results"]
    if len(results) != len(packages):
        raise RuntimeError("incomplete OSV response")
    findings = [
        {"package": p["name"], "version": p["version"], "vulnerabilities": r["vulns"]}
        for p, r in zip(packages, results) if r.get("vulns")
    ]
    print(f"Checked {len(packages)} locked dependencies against OSV")
    print(json.dumps(findings, indent=2))
    return bool(findings)


if __name__ == "__main__":
    raise SystemExit(main())
