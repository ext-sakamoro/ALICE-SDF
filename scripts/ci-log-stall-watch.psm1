# Runs a command with its output in a log file and stops it when the log stops growing.
# Author: Moroya Sakamoto
#
# A build that hangs without output otherwise holds the self-hosted runner until the job's
# timeout-minutes, and the job log only says "The operation was canceled". Here the command's
# whole process tree is killed after StallSeconds without a new byte on stdout or stderr, and
# the caller gets Hung = $true and the last line printed, so the failure names the hang and
# where it stopped.
#
#   Import-Module ./scripts/ci-log-stall-watch.psm1
#   $r = Invoke-StallWatched -FilePath x.bat -ArgumentList @('a', 'b') -LogPath out.log -StallSeconds 1200
#   $r.Hung / $r.ExitCode / $r.LastLine / $r.StalledSeconds
#
# Tests: scripts/test-ci-log-stall-watch.ps1

function Get-LogSize([string[]] $Paths) {
    $total = 0
    foreach ($p in $Paths) {
        if (Test-Path -LiteralPath $p) { $total += (Get-Item -LiteralPath $p).Length }
    }
    return $total
}

function Get-LastLine([string[]] $Paths) {
    $last = ''
    foreach ($p in $Paths) {
        if (Test-Path -LiteralPath $p) {
            $l = Get-Content -LiteralPath $p -Tail 1 -ErrorAction SilentlyContinue
            if ($l) { $last = "$l" }
        }
    }
    return $last
}

function Invoke-StallWatched {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)] [string] $FilePath,
        [string[]] $ArgumentList = @(),
        # stdout goes here; stderr goes to "$LogPath.stderr" and is appended at the end
        [Parameter(Mandatory = $true)] [string] $LogPath,
        [Parameter(Mandatory = $true)] [ValidateRange(1, 86400)] [int] $StallSeconds,
        [ValidateRange(1, 3600)] [int] $PollSeconds = 5
    )
    $errPath = "$LogPath.stderr"
    foreach ($p in @($LogPath, $errPath)) {
        if (Test-Path -LiteralPath $p) { Remove-Item -LiteralPath $p -Force }
    }
    $start = @{
        FilePath               = $FilePath
        RedirectStandardOutput = $LogPath
        RedirectStandardError  = $errPath
        NoNewWindow            = $true
        PassThru               = $true
    }
    if ($ArgumentList.Count -gt 0) { $start.ArgumentList = $ArgumentList }
    $proc = Start-Process @start
    # read the handle now: without it ExitCode can stay empty after the process exits
    $null = $proc.Handle

    $size = -1
    $lastGrowth = [DateTime]::UtcNow
    $hung = $false
    $stalled = 0
    while (-not $proc.HasExited) {
        $proc.WaitForExit($PollSeconds * 1000) | Out-Null
        if ($proc.HasExited) { break }
        $now = Get-LogSize @($LogPath, $errPath)
        if ($now -ne $size) {
            $size = $now
            $lastGrowth = [DateTime]::UtcNow
            continue
        }
        $stalled = [int]([DateTime]::UtcNow - $lastGrowth).TotalSeconds
        if ($stalled -ge $StallSeconds) {
            $hung = $true
            break
        }
    }

    $lastLine = Get-LastLine @($LogPath, $errPath)
    if ($hung) {
        # the whole tree: the batch file's cmd, UAT, UBT and the compilers it started
        try { $proc.Kill($true) } catch { }
        $proc.WaitForExit(30000) | Out-Null
    }
    else {
        $proc.WaitForExit()
    }
    if (Test-Path -LiteralPath $errPath) {
        Get-Content -LiteralPath $errPath | Add-Content -LiteralPath $LogPath
        Remove-Item -LiteralPath $errPath -Force
    }
    $exit = if ($hung) { $null } else { $proc.ExitCode }
    return [pscustomobject]@{
        Hung           = $hung
        ExitCode       = $exit
        LastLine       = $lastLine
        StalledSeconds = $stalled
    }
}

Export-ModuleMember -Function Invoke-StallWatched
