[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Push-Location $Root
try {
    $env:PYTHONPATH = Join-Path $Root "src"

    $resolved = $RunDir
    if (-not (Test-Path -LiteralPath $resolved)) {
        $resolved = Join-Path $Root (Join-Path "runs" $RunDir)
    }
    if (-not (Test-Path -LiteralPath $resolved -PathType Container)) {
        [Console]::Error.WriteLine("Run directory not found: $RunDir")
        exit 12
    }

    & python src/audit_run.py --run-dir $resolved
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
