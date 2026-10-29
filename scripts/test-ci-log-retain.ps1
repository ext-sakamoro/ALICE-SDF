# Tests of scripts/ci-log-sanitize.psm1 and scripts/ci-log-retain.ps1 (invented names and paths).
# Author: Moroya Sakamoto
#
# Plants the account and machine names, the profile directory, drive and install roots, a UNC
# path, Unix home paths, the Unity serial / machine id / licence client channel and an e-mail in
# log lines, and fails when any of them survives the sanitizer, when a line that must stay
# (the failure class, the module name, a URL) is lost, or when nothing was compared.
#
#   pwsh -NoProfile -File scripts/test-ci-log-retain.ps1

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$env:USERNAME = "山田花子"
$env:COMPUTERNAME = "BUILDBOX-7"
$env:USERPROFILE = "C:\Users\山田花子"
Import-Module (Join-Path $PSScriptRoot "ci-log-sanitize.psm1") -Force

$failures = 0
$checks = 0
function Check([bool] $ok, [string] $what) {
    $script:checks++
    if (-not $ok) { $script:failures++; Write-Output "FAIL: $what" }
}

# an invented Unix home path; joined at run time so the public text check of this repository
# (which rejects a literal home path in a tracked file) does not read the fixture as one
$homeLine = 'home ' + '/' + 'Users/hanako/Library/Logs/x.log and ' + '/' + 'home/hanako/.cache/y'

# planted: input line, then the fragments that must not survive
$planted = @(
    @('shell: C:\Users\山田花子\AppData\Local\Microsoft\WindowsApps\pwsh.EXE -command ". ''{0}''"', @('山田花子', 'Users', 'WindowsApps', 'AppData')),
    @("Copying 'C:\Users\山田花子\.gitconfig' to 'E:\actions-runner\_work\_temp\0f3a\.gitconfig'", @('山田花子', 'actions-runner', '_temp', 'Users')),
    @('Using C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe', @('Program Files', 'Visual Studio', 'Community', 'Current\Bin')),
    @('Unreal Engine (D:\UE_9.9)	Running: D:\UE_9.9\Engine\Binaries\DotNET\UnrealBuildTool\UnrealBuildTool.dll', @('UE_9.9', 'Engine\Binaries', 'DotNET')),
    @('"path": "E:\\actions-runner\\_work\\R\\R\\ci-unity"', @('actions-runner', '_work', 'ci-unity')),
    @('lib E:/ci-work/ue5-123/plugin-out/HostProject/Plugins/X/x.lib built', @('ci-work', 'plugin-out', 'HostProject')),
    @('cache at \\buildhost\share\ddc\cache.bin', @('buildhost', 'share\ddc')),
    @($homeLine, @('hanako', 'Library/Logs', '.cache')),
    @('[Licensing::Module] Serial number assigned to: "F4-ABCD-EFGH-IJKL-MNOP-XXXX"', @('ABCD', 'MNOP')),
    @('  Machine Id:              qWeRt/YuIoP123+asdf=', @('qWeRt', 'asdf=')),
    @('Connection attempt to the License Client on channel: "LicenseClient-山田花子" failed', @('山田花子')),
    @('channel "LicenseClient-hanako-notifications" connected', @('hanako')),
    @('signed in as hanako.yamada@mail.invalid on BUILDBOX-7', @('hanako', 'mail.invalid', 'BUILDBOX-7')),
    @('E:\a\b.log failed, see C:\c\d for the rest', @('E:\', 'C:\', '\a\', '\c\'))
)
foreach ($p in $planted) {
    $out = Protect-CiLine $p[0]
    foreach ($frag in $p[1]) { Check (-not $out.Contains($frag)) "'$frag' survived in: $out" }
    Check ($out -notmatch '(?<![A-Za-z0-9])[A-Za-z]:[\\/]') "a drive path survived in: $out"
}

# kept: the failure class and module / rule names, the plain text, a URL, a file name
$kept = @(
    @("Expecting to find a type to be declared in a module rules named 'SampleToolset' in 'UE5Rules, Version=0.0.0.0'", @('SampleToolset', 'UE5Rules', 'module rules')),
    @('Result: Failed (RulesError)', @('Result: Failed (RulesError)')),
    @('see https://github.com/example-org/example-repo/actions for the run', @('https://github.com/example-org/example-repo/actions')),
    @('FAIL: BuildPlugin failed (see E:\ci-work\ue5-9\logs\buildplugin.log)', @('FAIL: BuildPlugin failed', 'buildplugin.log')),
    @('cache at \\buildhost\share\ddc\cache.bin', @('cache at', 'cache.bin')),
    @($homeLine, @('home', 'x.log', ' and '))
)
foreach ($k in $kept) {
    $out = Protect-CiLine $k[0]
    foreach ($frag in $k[1]) { Check $out.Contains($frag) "'$frag' was lost from: $out" }
}

# a line whose path the patterns cannot strip (a drive glued to a word) is withheld, not printed
$glued = Protect-CiLine 'abcC:\Users\someone\x'
Check ((-not $glued.Contains('C:\')) -and (-not $glued.Contains('someone'))) "a residual drive path was printed: $glued"
Check ((Protect-CiLine $null) -eq '') "a null line is not an empty string"

# ci-log-retain.ps1 end to end: copy, prune, sanitized summary
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("ci-log-retain-test-" + [guid]::NewGuid().ToString("N"))
try {
    $src = Join-Path $tmp "work\logs"
    $root = Join-Path $tmp "kept"
    New-Item -ItemType Directory -Force $src | Out-Null
    $log = Join-Path $src "buildplugin.log"
    Set-Content -LiteralPath $log -Encoding utf8 -Value @(
        "Creating makefile for UnrealEditor (no existing makefile)",
        "Expecting to find a type to be declared in a module rules named 'SampleToolset' in 'UE5Rules'",
        "Result: Failed (RulesError)",
        "Error: trace written to file C:\Users\山田花子\AppData\Local\UnrealBuildTool\Log.uba",
        "[Licensing::Module] Error: Serial number assigned to: ""F4-ABCD-EFGH-IJKL-MNOP-XXXX""",
        "an unrelated line"
    )
    $old = Join-Path $root "100-1"
    $recent = Join-Path $root "200-1"
    $other = Join-Path $root "notes"
    foreach ($d in @($old, $recent, $other)) { New-Item -ItemType Directory -Force $d | Out-Null }
    (Get-Item $old).LastWriteTime = (Get-Date).AddDays(-20)
    (Get-Item $other).LastWriteTime = (Get-Date).AddDays(-200)
    (Get-Item $recent).LastWriteTime = (Get-Date).AddDays(-3)

    $out = & pwsh -NoProfile -File (Join-Path $PSScriptRoot "ci-log-retain.ps1") -Path $src -Root $root -Run "300-2" -Job "ue5-1" 2>&1
    $code = $LASTEXITCODE
    $text = ($out | ForEach-Object { "$_" }) -join "`n"
    Check ($code -eq 0) "ci-log-retain.ps1 exited $code`n$text"
    $copy = Join-Path $root "300-2\ue5-1\buildplugin.log"
    Check (Test-Path -LiteralPath $copy) "the log was not kept at $copy"
    if (Test-Path -LiteralPath $copy) {
        Check ((Get-Content -LiteralPath $copy -Raw) -eq (Get-Content -LiteralPath $log -Raw)) "the kept log is not the original"
    }
    Check (-not (Test-Path -LiteralPath $old)) "a run directory older than 14 days was not pruned"
    Check (Test-Path -LiteralPath $recent) "a recent run directory was pruned"
    Check (Test-Path -LiteralPath $other) "a directory the script did not make was pruned"
    Check ($text.Contains("SampleToolset") -and $text.Contains("RulesError")) "the summary lost the failure:`n$text"
    Check ($text.Contains("4 matching line(s)")) "the summary did not read the planted lines:`n$text"
    Check ($text.Contains("1 older run(s) pruned")) "the summary does not report the prune:`n$text"
    foreach ($frag in @('山田花子', 'AppData', 'ABCD', 'MNOP', 'an unrelated line')) {
        Check (-not $text.Contains($frag)) "'$frag' is in the summary:`n$text"
    }
    Check ($text -notmatch '(?<![A-Za-z0-9])[A-Za-z]:[\\/]') "a drive path is in the summary:`n$text"

    # a missing source is not an error
    $out2 = & pwsh -NoProfile -File (Join-Path $PSScriptRoot "ci-log-retain.ps1") -Path (Join-Path $tmp "absent") -Root $root -Run "301-1" -Job "ue5-0" 2>&1
    Check ($LASTEXITCODE -eq 0) "a missing source failed the step: $out2"
    Check ((($out2 | ForEach-Object { "$_" }) -join "`n").Contains("0 file(s)")) "a missing source was not reported"
} finally {
    if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force }
}

Write-Output "ci-log-retain: $checks checks, $failures failed"
if ($checks -eq 0) { Write-Output "FAIL: nothing was compared"; exit 1 }
if ($failures -gt 0) { exit 1 }
exit 0
