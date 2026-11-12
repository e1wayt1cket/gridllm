[CmdletBinding()]
param(
    [ValidateSet("Unit", "Integration", "Full")]
    [string]$Level = "Full"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Push-Location $Root
try {
    $env:PYTHONPATH = Join-Path $Root "src"

    # File selection is the wrapper's definition of a level; pytest.ini still
    # owns marker selection and the `slow` exclusion.
    switch ($Level) {
        "Unit" {
            $targets = @(
                "tests/test_training_contract.py",
                "tests/test_frozen_guard.py",
                "tests/test_train_da_gate_guards.py",
                "tests/test_train_da_gate_resume.py"
            )
            $verbosity = "-q"
        }
        "Integration" {
            $targets = @(
                "tests/test_checkpoint_roundtrip.py",
                "tests/test_checkpoint_schema.py",
                "tests/test_da_quote_mapping.py",
                "tests/test_rl_b_c2_gate.py"
            )
            $verbosity = "-q"
        }
        "Full" {
            $targets = @("tests/")
            $verbosity = "-v"
        }
    }

    Write-Host "== pytest $Level =="
    & python -m pytest @targets $verbosity
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
