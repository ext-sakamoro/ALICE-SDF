# Keep a self-hosted job's logs on the runner and print a sanitized summary of them.
# Author: Moroya Sakamoto
#
# The editor / UAT / Unity logs name the machine (paths, account, licence fields), so they are
# not uploaded as public artifacts. Instead:
#   1. every file under -Path (files or directories) is copied to <Root>\<Run>\<Job>\
#   2. run directories under <Root> (named <run id>-<attempt>) older than -KeepDays are removed
#   3. the lines matching -Pattern (at most -MaxLines per file) are printed through
#      Protect-CiLine (scripts/ci-log-sanitize.psm1): the failure's class and the module / rule
#      it names stay, drive paths, profile directories, install roots and licence fields go
# A missing source is not an error (the step runs with if: always(), also when the job failed
# before writing a log); it says so and exits 0.
#
#   pwsh scripts/ci-log-retain.ps1 -Path E:\alice-ci\ue5-1\logs -Root E:\alice-ci-logs -Run 1-1 -Job ue5-0

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string[]] $Path,
    [Parameter(Mandatory = $true)] [string] $Root,
    # <run id>-<attempt>
    [Parameter(Mandatory = $true)] [ValidatePattern('^\d+-\d+$')] [string] $Run,
    [Parameter(Mandatory = $true)] [ValidatePattern('^[A-Za-z0-9_.\-]+$')] [string] $Job,
    [ValidateRange(1, 365)] [int] $KeepDays = 14,
    [string] $Pattern = 'RulesError|rules named|error|Error:|FAIL|Result:|BUILD |Exception|\[CiSmokeTest\]',
    [ValidateRange(1, 1000)] [int] $MaxLines = 60
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Import-Module (Join-Path $PSScriptRoot "ci-log-sanitize.psm1") -Force

$full = [System.IO.Path]::GetFullPath($Root)
if ($full -eq [System.IO.Path]::GetPathRoot($full)) { throw "-Root must not be a drive or file system root" }

$files = @()
foreach ($p in $Path) {
    if (Test-Path -LiteralPath $p -PathType Container) {
        $files += @(Get-ChildItem -LiteralPath $p -Recurse -File)
    } elseif (Test-Path -LiteralPath $p -PathType Leaf) {
        $files += @(Get-Item -LiteralPath $p)
    }
}

$dest = Join-Path (Join-Path $full $Run) $Job
if ($files.Count -gt 0) {
    New-Item -ItemType Directory -Force $dest | Out-Null
    foreach ($f in $files) { Copy-Item -LiteralPath $f.FullName (Join-Path $dest $f.Name) -Force }
}

# prune: only the <run id>-<attempt> directories this script makes, never anything else
$cutoff = (Get-Date).AddDays(-$KeepDays)
$pruned = 0
if (Test-Path -LiteralPath $full) {
    Get-ChildItem -LiteralPath $full -Directory |
        Where-Object { $_.Name -match '^\d+-\d+$' -and $_.Name -ne $Run -and $_.LastWriteTime -lt $cutoff } |
        ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force; $pruned++ }
}

Write-Output "logs: $($files.Count) file(s) kept on the runner for $KeepDays days (run $Run, job $Job); $pruned older run(s) pruned"
foreach ($f in $files) {
    $hits = @(Select-String -LiteralPath $f.FullName -Pattern $Pattern -ErrorAction SilentlyContinue)
    Write-Output ("== {0}: {1} matching line(s){2}" -f $f.Name, $hits.Count, $(if ($hits.Count -gt $MaxLines) { ", first $MaxLines" } else { "" }))
    $hits | Select-Object -First $MaxLines | ForEach-Object { "  " + (Protect-CiLine $_.Line) }
}
exit 0
