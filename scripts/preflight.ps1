[CmdletBinding()]
param(
    [string]$ConfigDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Push-Location $Root
try {
    $env:PYTHONPATH = Join-Path $Root "src"

    Write-Host "== environment =="
    & python -c "import sys, yaml, numpy, pandas, torch, gurobipy as gp; m = gp.Model(); m.Params.OutputFlag = 0; m.optimize(); print('python', sys.version.split()[0]); print('gurobi', tuple(gp.gurobi.version()))"
    if ($LASTEXITCODE -ne 0) {
        [Console]::Error.WriteLine("Python, dependency or Gurobi licence check failed")
        exit 10
    }

    Write-Host "== budget contract =="
    $contractArgs = @("src/training_contract.py", "--check")
    if ($ConfigDir -ne "") { $contractArgs += @("--config-dir", $ConfigDir) }
    & python @contractArgs
    if ($LASTEXITCODE -ne 0) {
        [Console]::Error.WriteLine("Training contract check failed")
        exit 11
    }

    Write-Host "== formal manifest and hashes =="
    $launchArgs = @("src/run_formal_training.py", "--check")
    if ($ConfigDir -ne "") { $launchArgs += @("--config-dir", $ConfigDir) }
    & python @launchArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
