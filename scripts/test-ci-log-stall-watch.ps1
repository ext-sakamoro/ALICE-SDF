# Tests of scripts/ci-log-stall-watch.psm1.
# Author: Moroya Sakamoto
#
# A command that keeps printing (stdout, or stderr only) runs to its own exit code; a command
# that prints and then goes silent is killed after StallSeconds with Hung set, its last line
# reported and nothing it would have printed later in the log. Fails when nothing was compared.
#
#   pwsh -NoProfile -File scripts/test-ci-log-stall-watch.ps1

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Import-Module (Join-Path $PSScriptRoot "ci-log-stall-watch.psm1") -Force

$failures = 0
$checks = 0
function Check([bool] $ok, [string] $what) {
    $script:checks++
    if (-not $ok) { $script:failures++; Write-Output "FAIL: $what" }
}

$pwshExe = (Get-Process -Id $PID).Path
$dir = Join-Path ([IO.Path]::GetTempPath()) ("stall-watch-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $dir | Out-Null

# -EncodedCommand: the script text crosses Start-Process's argument joining unchanged on every
# OS (quotes and ';' inside a -Command argument are re-split differently on Windows and Unix)
function Encode([string] $script) { [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($script)) }

function Run([string] $name, [string] $script, [int] $stall) {
    $log = Join-Path $dir "$name.log"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-StallWatched -FilePath $pwshExe -ArgumentList @('-NoProfile', '-NonInteractive', '-EncodedCommand', (Encode $script)) `
        -LogPath $log -StallSeconds $stall -PollSeconds 1
    $sw.Stop()
    $text = if (Test-Path $log) { Get-Content $log -Raw } else { '' }
    if ($null -eq $text) { $text = '' }
    return [pscustomobject]@{ R = $r; Text = $text; Seconds = $sw.Elapsed.TotalSeconds; Log = $log }
}

try {
    # 1. keeps printing on stdout for longer than the stall window, then exits 3
    $a = Run 'steady' "1..6 | ForEach-Object { 'tick' + `$_; Start-Sleep -Milliseconds 700 }; exit 3" 3
    Check (-not $a.R.Hung) "steady output was reported as a hang"
    Check ($a.R.ExitCode -eq 3) "steady: exit code $($a.R.ExitCode), want 3"
    Check ($a.Text -match 'tick6') "steady: the last line is missing from the log"

    # 2. prints only on stderr, longer than the stall window: still progress, exit 0
    $b = Run 'stderr' "1..6 | ForEach-Object { [Console]::Error.WriteLine('err' + `$_); Start-Sleep -Milliseconds 700 }; exit 0" 3
    Check (-not $b.R.Hung) "stderr-only output was reported as a hang"
    Check ($b.R.ExitCode -eq 0) "stderr: exit code $($b.R.ExitCode), want 0"
    Check ($b.Text -match 'err6') "stderr: the stderr lines were not appended to the log"
    Check (-not (Test-Path "$($b.Log).stderr")) "stderr: the side file was left behind"

    # 3. prints, then goes silent for 60 s: killed after ~2 s of silence
    $c = Run 'hang' "'started'; 'step 13 of 21'; Start-Sleep -Seconds 60; 'never printed'" 2
    Check ($c.R.Hung) "a silent command was not reported as a hang"
    Check ($null -eq $c.R.ExitCode) "hang: an exit code was reported for a killed command"
    Check ($c.R.LastLine -eq 'step 13 of 21') "hang: last line '$($c.R.LastLine)', want 'step 13 of 21'"
    Check ($c.R.StalledSeconds -ge 2) "hang: stalled $($c.R.StalledSeconds) s, want >= 2"
    Check ($c.Seconds -lt 30) "hang: took $([int]$c.Seconds) s, the command was not killed"
    Check ($c.Text -notmatch 'never printed') "hang: the command kept running after the kill"

    # 4. a silent child of the command is killed too (the whole tree)
    $marker = Join-Path $dir "child-alive.txt"
    $child = Encode "Start-Sleep -Seconds 6; Set-Content -LiteralPath '$marker' x"
    # the child alone writes the marker; the parent only waits on it
    $d = Run 'tree' "'parent'; & '$pwshExe' -NoProfile -NonInteractive -EncodedCommand $child" 2
    Check ($d.R.Hung) "tree: a silent child was not reported as a hang"
    Start-Sleep -Seconds 7
    Check (-not (Test-Path $marker)) "tree: the child process survived the kill"

    # 5. control for 4: the same child, not killed, does write the marker (else 4 proves nothing)
    $marker2 = Join-Path $dir "child-ran.txt"
    $child2 = Encode "Start-Sleep -Seconds 1; Set-Content -LiteralPath '$marker2' x"
    $e = Run 'tree-control' "'parent'; & '$pwshExe' -NoProfile -NonInteractive -EncodedCommand $child2" 30
    Check (-not $e.R.Hung) "tree control: reported as a hang"
    Check (Test-Path $marker2) "tree control: the child never wrote its marker, so case 4 compares nothing"
}
finally {
    Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
}

if ($checks -eq 0) { Write-Output "FAIL: nothing was compared"; exit 1 }
if ($failures -gt 0) { Write-Output "stall watch: $failures of $checks checks failed"; exit 1 }
Write-Output "stall watch: $checks checks passed"
exit 0
