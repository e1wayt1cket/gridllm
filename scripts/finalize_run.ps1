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

    $manifestPath = Join-Path $resolved "run_manifest.json"
    if (-not (Test-Path -LiteralPath $manifestPath)) {
        [Console]::Error.WriteLine("run_manifest.json missing; the run has not finished")
        exit 13
    }
    # A failed run still carries audit evidence, so only a run that has not
    # terminated is refused.
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $finished = $manifest.PSObject.Properties.Name -contains "finished_at"
    if (-not $finished -or [string]::IsNullOrWhiteSpace($manifest.finished_at)) {
        [Console]::Error.WriteLine("Run is not finished; refusing to finalize")
        exit 13
    }

    & python src/audit_run.py --run-dir $resolved --write
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
