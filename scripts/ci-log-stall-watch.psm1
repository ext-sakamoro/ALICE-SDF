# Runs a command with its output in log files and stops it when the output stops.
# Author: Moroya Sakamoto
#
# A build that hangs without output otherwise holds the self-hosted runner until the job's
# timeout-minutes, and the job log only says "The operation was canceled". Here the command's
# whole process tree is killed after StallSeconds without a new line on stdout or stderr, and
# the caller gets Hung = $true and the last line printed, so the failure names the hang and
# where it stopped.
#
#   Import-Module ./scripts/ci-log-stall-watch.psm1
#   $r = Invoke-StallWatched -FilePath x.bat -ArgumentList @('a', 'b') -LogPath out.log -StallSeconds 1200
#   $r.Hung / $r.ExitCode / $r.LastLine / $r.StalledSeconds / $r.StderrPath / $r.DroppedLines
#
# The process is started here (System.Diagnostics.Process with ArgumentList: each argument
# reaches the command as given) and its stdout / stderr lines are written to $LogPath and
# "$LogPath.stderr" by this module. A process the command leaves running (a detached grandchild,
# or a server such as mspdbsrv that a compiler starts outside the tree) can keep the pipes open
# after the command exits: the module waits a bounded time for the streams to end, then closes
# its files and counts any later line as dropped, so it neither waits on such a process nor
# fails because one still writes. Both files are opened shared, readable while they are written.
# -OnHang runs before the kill with the process object, so the caller can record what the tree
# is doing (its output goes to the host; its errors are reported, not thrown).
#
# Tests: scripts/test-ci-log-stall-watch.ps1

if (-not ('AliceCi.StallRunner' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Diagnostics;
using System.IO;
using System.Threading;

namespace AliceCi {
    public sealed class StallRunner : IDisposable {
        readonly object gate = new object();
        readonly ManualResetEventSlim outEnd = new ManualResetEventSlim(false);
        readonly ManualResetEventSlim errEnd = new ManualResetEventSlim(false);
        StreamWriter outW, errW;
        bool closed;
        long lines, dropped;
        string last = "";

        public Process Proc { get; private set; }
        public long Lines { get { lock (gate) return lines; } }
        public long Dropped { get { lock (gate) return dropped; } }
        public string LastLine { get { lock (gate) return last; } }

        static StreamWriter Open(string path) {
            var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.ReadWrite | FileShare.Delete);
            return new StreamWriter(fs) { AutoFlush = true };
        }

        public static StallRunner Start(string file, string[] args, string outPath, string errPath) {
            var r = new StallRunner();
            r.outW = Open(outPath);
            r.errW = Open(errPath);
            var psi = new ProcessStartInfo(file) {
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };
            foreach (var a in args) psi.ArgumentList.Add(a);
            var p = new Process { StartInfo = psi };
            p.OutputDataReceived += (s, e) => r.Take(r.outW, e.Data, r.outEnd);
            p.ErrorDataReceived += (s, e) => r.Take(r.errW, e.Data, r.errEnd);
            p.Start();
            p.BeginOutputReadLine();
            p.BeginErrorReadLine();
            r.Proc = p;
            return r;
        }

        void Take(StreamWriter w, string line, ManualResetEventSlim end) {
            if (line == null) { end.Set(); return; }
            lock (gate) {
                if (closed) { dropped++; return; }
                w.WriteLine(line);
                lines++;
                if (line.Trim().Length > 0) last = line;
            }
        }

        // true when both streams ended within the time (no process holds them any more)
        public bool WaitStreams(int milliseconds) {
            var deadline = DateTime.UtcNow.AddMilliseconds(milliseconds);
            if (!outEnd.Wait(milliseconds)) return false;
            var rest = (int)Math.Max(0, (deadline - DateTime.UtcNow).TotalMilliseconds);
            return errEnd.Wait(rest);
        }

        public void Close() {
            lock (gate) {
                if (closed) return;
                closed = true;
                outW.Dispose();
                errW.Dispose();
            }
        }

        public void Dispose() { Close(); }
    }
}
'@
}

# reads a file another process may still hold open for writing; $null when it cannot be read
function Read-Shared([string] $Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try {
        $fs = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete)
        try { return (New-Object IO.StreamReader($fs)).ReadToEnd() } finally { $fs.Dispose() }
    }
    catch { return $null }
}

function Invoke-StallWatched {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)] [string] $FilePath,
        [string[]] $ArgumentList = @(),
        # stdout goes here; stderr goes to "$LogPath.stderr" (both replaced at the start)
        [Parameter(Mandatory = $true)] [string] $LogPath,
        [Parameter(Mandatory = $true)] [ValidateRange(1, 86400)] [int] $StallSeconds,
        [ValidateRange(1, 3600)] [int] $PollSeconds = 5,
        # how long to wait, after the command exits, for the output streams to end
        [ValidateRange(0, 600)] [int] $DrainSeconds = 5,
        [scriptblock] $OnHang = $null
    )
    $errPath = "$LogPath.stderr"
    $runner = [AliceCi.StallRunner]::Start($FilePath, [string[]] $ArgumentList, $LogPath, $errPath)
    $proc = $runner.Proc
    try {
        $seen = -1
        $lastGrowth = [DateTime]::UtcNow
        $hung = $false
        $stalled = 0
        while (-not $proc.WaitForExit($PollSeconds * 1000)) {
            $now = $runner.Lines
            if ($now -ne $seen) {
                $seen = $now
                $lastGrowth = [DateTime]::UtcNow
                continue
            }
            $stalled = [int]([DateTime]::UtcNow - $lastGrowth).TotalSeconds
            if ($stalled -ge $StallSeconds) {
                $hung = $true
                break
            }
        }
        $lastLine = $runner.LastLine
        if ($hung) {
            if ($OnHang) {
                # to the host, not the pipeline: the function's only output is the result object
                try { & $OnHang $proc | Out-Host } catch { Write-Host "stall watch: OnHang failed: $($_.Exception.Message)" }
            }
            # the whole tree: the batch file's cmd, UAT, UBT and the compilers it started
            try { $proc.Kill($true) } catch { }
            $proc.WaitForExit(30000) | Out-Null
        }
        $drained = $runner.WaitStreams($DrainSeconds * 1000)
        $exit = if ($hung) { $null } else { $proc.ExitCode }
    }
    finally {
        $runner.Close()
    }
    return [pscustomobject]@{
        Hung           = $hung
        ExitCode       = $exit
        LastLine       = $lastLine
        StalledSeconds = $stalled
        StderrPath     = $errPath
        # false when a process outside the command still held its output after it exited
        Drained        = $drained
        DroppedLines   = $runner.Dropped
    }
}

Export-ModuleMember -Function Invoke-StallWatched, Read-Shared
