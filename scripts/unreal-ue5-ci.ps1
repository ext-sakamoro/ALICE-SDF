# Unreal Engine 5 compatibility gate for the ALICE-SDF plugin.
# Author: Moroya Sakamoto
#
# Runs, against a real installed engine, everything `cargo build --features
# unreal` does not: the C++ plugin compiles and links (UAT BuildPlugin, editor
# + game targets), every .usf / .ush compiles (global shaders are compiled
# when the editor starts, a failure is fatal), and the two automation tests
# in Source/AliceSDF/Private/Tests run on the editor's real RHI:
#   AliceSDF.Unreal.FfiCorpusParity  — alice_sdf.dll vs Rust golden, bit-exact
#   AliceSDF.Unreal.HlslGpuOracle    — transpiled HLSL on the GPU vs the DLL
# (see examples/unreal_corpus_oracle.rs for the inputs).
#
# The GitHub-hosted runners have no Unreal; .github/workflows/ci.yml runs this
# on the self-hosted `ue5` Windows runner. It is the same script locally:
#
#   pwsh scripts/unreal-ue5-ci.ps1 -EngineRoot E:\UE_5.7 -WorkDir E:\alice-ci\ue5
#
# Steps:
#   1. cargo build --release --features unreal   → alice_sdf.dll / .lib
#   2. cargo run --example unreal_corpus_oracle  → golden dir + generated shaders,
#      and `git diff --exit-code` on the committed generated files (drift gate)
#   3. stage unreal-plugin/ with the fresh DLL + include/alice_sdf.h
#   4. RunUAT BuildPlugin -TargetPlatforms=Win64 -Rocket
#   5. host project + UnrealEditor-Cmd -ExecCmds="Automation RunTests AliceSDF.Unreal"
#   6. parse the automation report (index.json): every test must pass
#
# Exit code 0 only when all six pass. Logs land in $WorkDir\logs.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $EngineRoot,
    [Parameter(Mandatory = $true)] [string] $WorkDir,
    # The repository checkout (defaults to the script's parent).
    [string] $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    # Skip the cargo steps when the DLL was already built (local iteration).
    [switch] $SkipCargo
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Step([string] $Msg) { Write-Host "`n== $Msg" -ForegroundColor Cyan }
function Fail([string] $Msg) { Write-Host "FAIL: $Msg" -ForegroundColor Red; exit 1 }

if (-not (Test-Path (Join-Path $EngineRoot "Engine\Build\BatchFiles\RunUAT.bat"))) {
    Fail "no engine at $EngineRoot (Engine\Build\BatchFiles\RunUAT.bat missing)"
}
$BuildVersion = Get-Content (Join-Path $EngineRoot "Engine\Build\Build.version") | ConvertFrom-Json
$EngineAssoc = "$($BuildVersion.MajorVersion).$($BuildVersion.MinorVersion)"
Write-Host "engine $EngineAssoc.$($BuildVersion.PatchVersion) at $EngineRoot"

$Logs = Join-Path $WorkDir "logs"
$Golden = Join-Path $WorkDir "golden"
$Staged = Join-Path $WorkDir "plugin-src"
$Packaged = Join-Path $WorkDir "plugin-out"
$Host_ = Join-Path $WorkDir "host"
$Report = Join-Path $WorkDir "report"
foreach ($d in @($Logs, $Golden)) { New-Item -ItemType Directory -Force $d | Out-Null }

# ── 1. cargo build (cdylib, unreal feature) ────────────────────────────────
$TargetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $RepoRoot "target" }
$Dll = Join-Path $TargetDir "release\alice_sdf.dll"
$Lib = Join-Path $TargetDir "release\alice_sdf.dll.lib"
if (-not $SkipCargo) {
    Step "cargo build --release --features unreal"
    Push-Location $RepoRoot
    try {
        cargo build --release --features unreal
        if ($LASTEXITCODE -ne 0) { Fail "cargo build" }

        # ── 2. corpus oracle inputs + generated-file drift gate ────────────
        Step "cargo run --example unreal_corpus_oracle (golden → $Golden)"
        # --features unreal (not just hlsl): a different feature set rebuilds
        # the cdylib in the same target dir, and the alice_sdf.dll staged for
        # Unreal would silently lose every extern "C" fn (128 LNK2019).
        cargo run --release --example unreal_corpus_oracle --features unreal -- --plugin unreal-plugin --golden $Golden
        if ($LASTEXITCODE -ne 0) { Fail "unreal_corpus_oracle" }
        git diff --exit-code --stat -- unreal-plugin/Shaders/CorpusOracle unreal-plugin/Source/AliceSDF/Private/Generated
        if ($LASTEXITCODE -ne 0) {
            Fail "generated corpus files are stale — run the example and commit unreal-plugin/Shaders/CorpusOracle + Private/Generated"
        }
        $Untracked = git ls-files --others --exclude-standard -- unreal-plugin/Shaders/CorpusOracle unreal-plugin/Source/AliceSDF/Private/Generated
        if ($Untracked) { Fail "generated corpus files not committed:`n$Untracked" }
    } finally { Pop-Location }
}
foreach ($f in @($Dll, $Lib)) { if (-not (Test-Path $f)) { Fail "missing $f (cargo build --release --features unreal)" } }
# Snapshot the binaries: any later cargo command with a different feature set
# (`--features image` for the icon, a plain `cargo test`, ...) rewrites
# target/release/alice_sdf.dll in place, and staging would pick up a library
# without the FFI. Everything below uses the copy.
$LibDirWork = Join-Path $WorkDir "lib"
New-Item -ItemType Directory -Force $LibDirWork | Out-Null
Copy-Item $Dll (Join-Path $LibDirWork "alice_sdf.dll") -Force
Copy-Item $Lib (Join-Path $LibDirWork "alice_sdf.lib") -Force
$Dll = Join-Path $LibDirWork "alice_sdf.dll"
$Lib = Join-Path $LibDirWork "alice_sdf.lib"
# The FFI is what the plugin links; a cdylib built with a narrower feature
# set exports none of it and only fails 20 minutes later, in the linker.
$DumpBin = Get-ChildItem "${env:ProgramFiles}\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*in\Hostx64d\dumpbin.exe" -ErrorAction SilentlyContinue | Select-Object -Last 1
if ($DumpBin) {
    $Exports = & $DumpBin.FullName -exports $Dll | Select-String -Pattern "alice_sdf_[a-z0-9_]+" -AllMatches
    $Count = ($Exports.Matches.Value | Sort-Object -Unique).Count
    if ($Count -lt 100) { Fail "$Dll exports only $Count alice_sdf_* symbols — it was built without the unreal / ffi features" }
    Write-Host "cdylib exports $Count alice_sdf_* symbols"
} else {
    Write-Host "dumpbin not found — skipping the export count check"
}
if ((Get-ChildItem $Golden -Filter *.asdf).Count -eq 0) { Fail "no .asdf in $Golden (run the corpus oracle example)" }
# -SkipCargo keeps whatever golden the work dir already has. A golden older
# than the library is a golden from a different law: the FFI test would then
# report a "drift" that is only the stale file (3.2.0, after the smooth-law
# unification).
$GoldenStamp = Join-Path $Golden "points.txt"
if ((Get-Item $GoldenStamp).LastWriteTimeUtc -lt (Get-Item $Dll).LastWriteTimeUtc) {
    Fail "$Golden is older than $Dll — regenerate it (cargo run --release --example unreal_corpus_oracle --features unreal -- --plugin unreal-plugin --golden $Golden)"
}

# ── 3. stage the plugin with the fresh binary + header ─────────────────────
Step "stage plugin → $Staged"
foreach ($d in @($Staged, $Packaged, $Host_, $Report)) {
    if (Test-Path $d) { Remove-Item -Recurse -Force $d }
}
Copy-Item -Recurse (Join-Path $RepoRoot "unreal-plugin") $Staged
$Win64 = Join-Path $Staged "ThirdParty\AliceSDF\lib\Win64"
Copy-Item $Dll (Join-Path $Win64 "alice_sdf.dll") -Force
Copy-Item $Lib (Join-Path $Win64 "alice_sdf.lib") -Force
Copy-Item (Join-Path $RepoRoot "include\alice_sdf.h") (Join-Path $Staged "ThirdParty\AliceSDF\include\alice_sdf.h") -Force
# HostProject~ is documentation for humans; UAT makes its own.
if (Test-Path (Join-Path $Staged "HostProject~")) { Remove-Item -Recurse -Force (Join-Path $Staged "HostProject~") }

# ── 4. UAT BuildPlugin (UnrealEditor + UnrealGame, Win64) ──────────────────
Step "RunUAT BuildPlugin (Win64, editor + game targets)"
$UatLog = Join-Path $Logs "buildplugin.log"
& (Join-Path $EngineRoot "Engine\Build\BatchFiles\RunUAT.bat") BuildPlugin `
    -Plugin="$Staged\AliceSDF.uplugin" -Package="$Packaged" -TargetPlatforms=Win64 -Rocket 2>&1 |
    Tee-Object -FilePath $UatLog | Select-String -Pattern "error|warning C|Result:|BUILD " | ForEach-Object { $_.Line }
if ($LASTEXITCODE -ne 0) { Fail "BuildPlugin failed (see $UatLog)" }
if (-not (Test-Path (Join-Path $Packaged "Binaries\Win64\UnrealEditor-AliceSDF.dll"))) {
    Fail "BuildPlugin produced no UnrealEditor-AliceSDF.dll"
}
# The packaged plugin must carry the native library, or every install of it
# fails at module load (error 126, or a delay-load fault on the first call).
# AliceSDF.Build.cs stages it next to the module; this is the gate.
# UAT's plugin filter keeps the .uplugin, build products, /Binaries/ThirdParty,
# /Resources, /Content, /Shaders and /Source; /ThirdParty is dropped unless
# Config/FilterPlugin.ini names it. Both places are acceptable — the module
# looks in Binaries\Win64 first, then ThirdParty (FAliceSdfModule).
$PackagedDll = Get-ChildItem $Packaged -Recurse -Filter alice_sdf.dll -File | Select-Object -First 1
if (-not $PackagedDll) {
    Fail "packaged plugin carries no alice_sdf.dll — the module cannot load (GetLastError=126). Check Config/FilterPlugin.ini and RuntimeDependencies in AliceSDF.Build.cs"
}
if ((Get-FileHash $PackagedDll.FullName).Hash -ne (Get-FileHash $Dll).Hash) {
    Fail "packaged alice_sdf.dll ($($PackagedDll.FullName)) is not the one just built"
}
Write-Host "packaged native library: $($PackagedDll.FullName.Substring($Packaged.Length + 1))"

# ── 5. host project + automation run ───────────────────────────────────────
Step "host project + Automation RunTests AliceSDF.Unreal"
New-Item -ItemType Directory -Force (Join-Path $Host_ "Plugins") | Out-Null
Copy-Item -Recurse $Packaged (Join-Path $Host_ "Plugins\AliceSDF")
$Uproject = Join-Path $Host_ "AliceSdfHost.uproject"
@{
    FileVersion       = 3
    EngineAssociation = $EngineAssoc
    Category          = ""
    Description       = "ALICE-SDF plugin CI host (generated by scripts/unreal-ue5-ci.ps1)"
    Plugins           = @(
        @{ Name = "AliceSDF"; Enabled = $true },
        # The sample material is built by an editor Python script.
        @{ Name = "PythonScriptPlugin"; Enabled = $true }
    )
} | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 $Uproject

$EditorLog = Join-Path $Logs "editor.log"
# DDC: an installed (Launcher) engine's writable local cache is ZenLocal, and
# its `Local` FileSystem node is DeleteOnly — so when zenserver.exe fails to
# start there is no writable backend and the editor exits with a fatal
# (DerivedDataBackends.cpp). On this runner it fails every time: the Zen data
# dir sits under a non-ASCII user profile and the installed zenserver is older
# than the engine. `-ddc=NoZenLocalFallback` is Epic's own graph for that case
# (a writable FileSystem Local), and the cache goes on the work drive.
$Ddc = Join-Path $WorkDir "ddc"
New-Item -ItemType Directory -Force $Ddc | Out-Null
[Environment]::SetEnvironmentVariable("UE-LocalDataCachePath", $Ddc, "Process")
$env:ALICE_SDF_GOLDEN_DIR = $Golden
$env:ALICE_SDF_CORPUS_ORACLE = "1"
$env:ALICE_SDF_REQUIRE_GOLDEN = "1"
$env:ALICE_SDF_REQUIRE_GPU = "1"
$EditorCmd = Join-Path $EngineRoot "Engine\Binaries\Win64\UnrealEditor-Cmd.exe"
# -unattended: no dialogs; -NoCrashReporter: a fatal (e.g. a .usf that does
# not compile) exits non-zero instead of waiting on the crash window.
& $EditorCmd $Uproject `
    -ddc=NoZenLocalFallback `
    -ExecCmds="Automation RunTests AliceSDF.Unreal;Quit" `
    -ReportExportPath="$Report" -abslog="$EditorLog" `
    -unattended -nop4 -nosplash -NoSound -NoCrashReporter -stdout -FullStdOutLogOutput 2>&1 |
    Select-String -Pattern "LogAutomation|AliceSDF|Shader.*(error|failed)|Fatal|Error:" | ForEach-Object { $_.Line }
$EditorExit = $LASTEXITCODE
if ($EditorExit -ne 0) { Fail "UnrealEditor-Cmd exited $EditorExit (see $EditorLog)" }

# ── 6. the report ──────────────────────────────────────────────────────────
Step "automation report"
$Index = Join-Path $Report "index.json"
if (-not (Test-Path $Index)) { Fail "no automation report at $Index (did the editor reach the tests?)" }
$Json = Get-Content $Index -Raw | ConvertFrom-Json
$Expected = @("AliceSDF.Unreal.FfiCorpusParity", "AliceSDF.Unreal.HlslGpuOracle")
$Failed = 0
foreach ($t in $Json.tests) {
    Write-Host ("{0,-40} {1}" -f $t.fullTestPath, $t.state)
    foreach ($e in $t.entries) {
        $ev = $e.event
        if ($ev.type -ne "Info" -or $ev.message -match "corpus|oracle|alice_sdf.dll") {
            Write-Host ("    [{0}] {1}" -f $ev.type, $ev.message)
        }
    }
    if ($t.state -ne "Success") { $Failed++ }
}
foreach ($name in $Expected) {
    if (-not ($Json.tests | Where-Object { $_.fullTestPath -eq $name })) {
        Write-Host "missing test: $name" -ForegroundColor Red
        $Failed++
    }
}
if ($Failed -gt 0) { Fail "$Failed automation test(s) failed or missing" }
# ── 7. the sample material ─────────────────────────────────────────────────
# Content/Python/create_alice_sdf_sample_material.py is what the docs tell a
# user to run, and Shaders/Public/AliceSdfSample.ush is what it compiles. A
# material that fails to build here fails in their project too.
Step "sample material (Content/Python/create_alice_sdf_sample_material.py)"
$MaterialScript = Join-Path $Host_ "Plugins\AliceSDF\Content\Python\create_alice_sdf_sample_material.py"
if (-not (Test-Path $MaterialScript)) { Fail "the packaged plugin has no $MaterialScript" }
$MaterialLog = Join-Path $Logs "material.log"
& $EditorCmd $Uproject `
    -ddc=NoZenLocalFallback -run=pythonscript -script="$MaterialScript" `
    -abslog="$MaterialLog" `
    -unattended -nop4 -nosplash -NoSound -NoCrashReporter -stdout |
    Select-String -Pattern "ALICE-SDF:|LogPython|Material|error|Error:" | ForEach-Object { $_.Line }
if ($LASTEXITCODE -ne 0) { Fail "the sample material script exited $LASTEXITCODE (see $MaterialLog)" }
$MaterialAsset = Join-Path $Host_ "Content\AliceSDF\M_AliceSDF_Sample.uasset"
if (-not (Test-Path $MaterialAsset)) { Fail "no M_AliceSDF_Sample.uasset was written (see $MaterialLog)" }
$ShaderErrors = Select-String -Path $MaterialLog -Pattern "Failed to compile Material|Shader compile error|error X[0-9]+" -ErrorAction SilentlyContinue
if ($ShaderErrors) { Fail "the sample material did not compile:`n$($ShaderErrors[0].Line)" }
Write-Host "sample material built: $([int]((Get-Item $MaterialAsset).Length / 1KB)) KB"

Write-Host "`nUE ${EngineAssoc}: plugin built, shaders compiled, $($Json.tests.Count) automation tests passed, sample material built" -ForegroundColor Green
exit 0
