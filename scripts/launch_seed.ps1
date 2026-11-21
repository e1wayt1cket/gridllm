[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [int]$Seed
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Push-Location $Root
try {
    $env:PYTHONPATH = Join-Path $Root "src"

    Write-Host "== preflight =="
    & (Join-Path $PSScriptRoot "preflight.ps1")
    if ($LASTEXITCODE -ne 0) {
        [Console]::Error.WriteLine("Preflight refused; seed $Seed was not started")
        exit $LASTEXITCODE
    }

    Write-Host "== launch seed $Seed =="
    & python src/run_formal_training.py --seed $Seed
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
