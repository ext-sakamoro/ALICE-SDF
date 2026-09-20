# Local pre-push check of the Unity side of vrchat-package (the part cargo and
# dotnet cannot see): shader compilation, UdonSharp compilation, sample scene
# generation, Kit product build. Mirrors the `unity-vrchat` job of
# .github/workflows/ci.yml stage by stage; the job runs the same
# AliceSDF.Editor.AliceSDF_CiChecks.RunBatch entry.
#
#   pwsh scripts/unity-preflight.ps1 [-Project <VRChat Worlds project>]
#        [-Unity <Unity.exe>] [-Stages setup,compile,samples,scenes,kit,verify,android]
#        [-SkipDrift] [-DriftOnly]
#
# The project must be a VRChat Worlds project (SDK 3.7+) whose
# Packages/manifest.json links this checkout as
#   "com.alice.sdf": "file:<path to vrchat-package>"
# Defaults are this PC's (MochiProductTest, Unity 2022.3.22f1); override on
# another machine. Exit 0 = every stage passed, 1 = a stage failed (its log is
# named), 2 = setup problem.
[CmdletBinding()]
param(
    [string]$Project = "$env:LOCALAPPDATA\VRChatCreatorCompanion\VRChatProjects\MochiProductTest",
    [string]$Unity = "E:\2022.3.22f1\Editor\Unity.exe",
    [string[]]$Stages = @("setup", "compile", "samples", "scenes", "kit", "verify", "android"),
    [switch]$SkipDrift,
    [switch]$DriftOnly
)
$ErrorActionPreference = "Stop"
# -File passes "a,b" as one string; accept both forms
$Stages = @($Stages | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$root = Split-Path -Parent $PSScriptRoot
$pkg = Join-Path $root "vrchat-package"

# --- drift: Samples~ (the source) vs Assets/Samples copies in a developer project
# Unity compiles the Assets/Samples/... copy of a sample, not Samples~, so an
# edit to Samples~ that was not synced (or the reverse) ships a stale sample.
function Test-SampleDrift([string]$proj) {
    $copies = Join-Path $proj "Assets\Samples\ALICE-SDF for VRChat"
    if (-not (Test-Path $copies)) { return 0 }
    $bad = 0
    $seen = 0
    foreach ($ver in Get-ChildItem $copies -Directory) {
        foreach ($sample in Get-ChildItem $ver.FullName -Directory) {
            $name = $sample.Name -replace '^SDF Gallery - ', ''
            $src = Join-Path $pkg "Samples~\SDF Gallery\Sample$name"
            if (-not (Test-Path $src)) { continue }
            foreach ($f in Get-ChildItem $src -File -Recurse | Where-Object { $_.Extension -in ".cs", ".shader", ".cginc", ".asset", ".json" }) {
                $rel = $f.FullName.Substring($src.Length + 1)
                $copy = Join-Path $sample.FullName $rel
                $seen++
                if (-not (Test-Path $copy)) { Write-Host "drift: $name/$rel missing in $($sample.FullName)"; $bad++; continue }
                if ((Get-FileHash $f.FullName).Hash -ne (Get-FileHash $copy).Hash) { Write-Host "drift: $name/$rel differs from Samples~ (run sync-samples / re-import the sample)"; $bad++ }
            }
        }
    }
    Write-Host "drift check: $seen sample file(s) compared in $proj"
    return $bad
}

if (-not (Test-Path $Unity)) { Write-Host "unity-preflight: Unity not found at $Unity (-Unity)"; exit 2 }
if (-not (Test-Path (Join-Path $Project "Packages\manifest.json"))) { Write-Host "unity-preflight: not a Unity project: $Project (-Project)"; exit 2 }
$manifest = Get-Content (Join-Path $Project "Packages\manifest.json") -Raw
if ($manifest -notmatch '"com\.alice\.sdf"\s*:\s*"file:') { Write-Host "unity-preflight: $Project does not link com.alice.sdf as a file: dependency"; exit 2 }
# the Creator Companion records the SDK in vpm-manifest.json, a hand-made project in manifest.json
$vpm = Join-Path $Project "Packages\vpm-manifest.json"
$sdk = $manifest -match 'com\.vrchat\.worlds' -or ((Test-Path $vpm) -and ((Get-Content $vpm -Raw) -match 'com\.vrchat\.worlds'))
if (-not $sdk) { Write-Host "unity-preflight: $Project has no com.vrchat.worlds (VRChat Worlds project required)"; exit 2 }

if (-not $SkipDrift) {
    $drift = 0
    foreach ($devProj in @((Join-Path $env:LOCALAPPDATA "VRChatCreatorCompanion\VRChatProjects\Mochi"), $Project)) { $drift += Test-SampleDrift $devProj }
    if ($drift -gt 0) { Write-Host "unity-preflight: $drift sample file(s) drifted between Samples~ and Assets/Samples"; exit 1 }
    Write-Host "unity-preflight: no Samples~ drift"
}
if ($DriftOnly) { exit 0 }

$logDir = Join-Path $root "target\unity-preflight"
New-Item -ItemType Directory -Force $logDir | Out-Null
$marker = Join-Path $Project "Library\alice_ci_kit.txt"
if (Test-Path $marker) { Remove-Item $marker }
# A run that imports the samples starts from nothing, like CI: the generator
# skips scenes that exist and the Kit builder keeps unchanged files
if ($Stages -contains "samples") {
    foreach ($gen in @("Assets\Samples", "Assets\AliceSDF_SampleScenes", "Assets\AliceSDFKit", "Assets\SerializedUdonPrograms")) {
        foreach ($path in @((Join-Path $Project $gen), (Join-Path $Project ($gen + ".meta")))) {
            if (Test-Path $path) { Remove-Item -Recurse -Force $path }
        }
    }
    Write-Host "unity-preflight: cleared generated folders in $Project"
}

function Invoke-Stage([string]$stage, [string]$target, [string]$logName) {
    $log = Join-Path $logDir "$logName.log"
    $env:ALICE_CI_STAGE = $stage
    $args = @("-batchmode", "-nographics", "-quit", "-projectPath", $Project, "-buildTarget", $target,
              "-executeMethod", "AliceSDF.Editor.AliceSDF_CiChecks.RunBatch", "-logFile", $log)
    # the working directory must be the Editor folder or the shader compiler
    # cannot find HLSLSupport.cginc (every shader then fails)
    $p = Start-Process -FilePath $Unity -ArgumentList $args -WorkingDirectory (Split-Path -Parent $Unity) -Wait -PassThru -NoNewWindow
    Get-Content $log | Where-Object { $_ -match '\[ALICE-CI\]|error CS|Shader error' } | ForEach-Object { Write-Host "  $_" }
    if ($p.ExitCode -ne 0) { Write-Host "unity-preflight: stage $stage FAILED (exit $($p.ExitCode)), log $log"; exit 1 }
}

foreach ($stage in $Stages) {
    Write-Host "== stage $stage"
    if ($stage -eq "kit") {
        # the product build spans domain reloads: run until the marker says done
        for ($i = 1; $i -le 6; $i++) {
            Invoke-Stage "kit" "Win64" "kit$i"
            $state = if (Test-Path $marker) { (Get-Content $marker -Raw).Trim() } else { "" }
            if ($state -like "done*") { Write-Host "  kit: $state"; break }
            if ($state -eq "failed") { Write-Host "unity-preflight: kit build failed"; exit 1 }
            if ($i -eq 6) { Write-Host "unity-preflight: kit build did not finish in 6 runs"; exit 1 }
        }
    } elseif ($stage -eq "android") {
        Invoke-Stage "android" "Android" "android"
    } else {
        Invoke-Stage $stage "Win64" $stage
    }
}
Write-Host "unity-preflight: all stages passed ($($Stages -join ', '))"
exit 0
