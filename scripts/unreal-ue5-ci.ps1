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
#   pwsh scripts/unreal-ue5-ci.ps1 -EngineRoot E:\UE_5.7 -WorkDir E:\ci-work\ue5
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

# DL / 更新途中の engine を「ある」と誤認しないよう 4 点を見る Engine\Build\BatchFiles
# は早い段階で展開されるので RunUAT.bat 単独では不十分 (2026-09-27 に Epic Launcher の
# DL 途中 29.5 GB 時点で RunUAT.bat だけ存在する状態を実測した engine 不在なら 40 秒で
# 落ちるが、不完全な engine を通すと UBT / shader compile の段で真因の分かりにくい red
# になる) editor 本体は DL 後半に来るので完了の代理指標になる
$RequiredEngineFiles = @(
    "Engine\Build\BatchFiles\RunUAT.bat",
    "Engine\Build\Build.version",
    "Engine\Binaries\Win64\UnrealEditor.exe",
    "Engine\Binaries\Win64\UnrealEditor-Cmd.exe"
)
$MissingEngineFiles = $RequiredEngineFiles | Where-Object { -not (Test-Path (Join-Path $EngineRoot $_)) }
if ($MissingEngineFiles) {
    Fail "engine at $EngineRoot is absent or incomplete (missing: $($MissingEngineFiles -join ', '))"
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
# An editor left open on the previous host project (a developer's, or one
# started to serve MCP) keeps its DerivedDataCache files locked and the
# cleanup below fails half way. Say so instead of dying inside Remove-Item.
$Stale = Get-Process UnrealEditor, UnrealEditor-Cmd -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -and $_.Path.StartsWith($EngineRoot, [StringComparison]::OrdinalIgnoreCase) }
if ($Stale) {
    Fail ("an Unreal editor from $EngineRoot is still running (PID " +
        (($Stale | ForEach-Object { $_.Id }) -join ", ") +
        ") and holds files under $WorkDir — close it and run again")
}
foreach ($d in @($Staged, $Packaged, $Host_, $Report)) {
    if (Test-Path $d) {
        try { Remove-Item -Recurse -Force $d -ErrorAction Stop }
        catch { Fail "could not clear $d ($($_.Exception.Message)). Something still holds a file there." }
    }
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
# The compile on this runner has stopped part-way (the cl.exe processes alive at ~0 CPU on an
# idle machine, no line printed) until timeout-minutes cancelled the job, where a good run
# compiles the plugin's 21 actions in ~13 min. It stopped with the Unreal Build Accelerator
# (UBA) and again with UBA off (UBT's parallel executor), so UBA is not the cause; the cause is
# not known yet. What this step does about it:
#  - UBA stays off (it was not the cause; the parallel executor keeps the compile in plain
#    cl.exe processes, which are easier to inspect). BuildPlugin has no switch that reaches UBT,
#    so the setting goes through UBT's environment config (UnrealBuildTool_<Category>__<Field>,
#    written by UBT to UnrealBuildTool.Env.BuildConfiguration.xml on every run; the same as
#    -NoUBA). UE 5.7 then compiles on its parallel executor; UE 5.8 always runs the UBA
#    executor and the setting turns its detouring off. The check below reads that file back, so
#    it fails if the setting did not reach UBT, whichever executor the engine names.
#  - the compilers' PDB server (mspdbsrv) gets an endpoint of this job's own, so a server left
#    by an earlier job (one that was killed or timed out) is not shared, and the servers this
#    job started are stopped at the end.
#  - the build processes already running before the build, the process tree at a hang and any
#    process still holding the output after BuildPlugin exits are printed: the evidence for the
#    cause.
#  - a build that goes silent for $StallMinutes min is stopped and fails with a HANG line and the
#    last line it printed, well before timeout-minutes (no single silent stretch of a good build
#    is near 20 minutes; the whole 21-action compile is ~13).
$env:UnrealBuildTool_BuildConfiguration__bAllowUBAExecutor = "false"
$env:_MSPDBSRV_ENDPOINT_ = "alicesdf-" + [guid]::NewGuid().ToString("N")
$StallMinutes = 20
$BuildStart = Get-Date
$BuildProcs = "cl", "link", "mspdbsrv", "UbaHost", "UnrealBuildTool", "ShaderCompileWorker", "ispc", "dotnet", "MSBuild", "UnrealEditor-Cmd"

# name, pid, parent, start time and CPU seconds of the build processes, and whether each one is
# inside the tree of $Root (when given)
function Show-BuildProcesses([string] $Title, $Root = $null) {
    $all = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    $inTree = @{}
    if ($Root) {
        $inTree[[int]$Root.Id] = $true
        $grew = $true
        while ($grew) {
            $grew = $false
            foreach ($q in $all) {
                if ($inTree.ContainsKey([int]$q.ParentProcessId) -and -not $inTree.ContainsKey([int]$q.ProcessId)) {
                    $inTree[[int]$q.ProcessId] = $true
                    $grew = $true
                }
            }
        }
    }
    $rows = @(Get-Process -Name $BuildProcs -ErrorAction SilentlyContinue)
    Write-Host "== processes: $Title ($($rows.Count))"
    foreach ($r in $rows) {
        $parent = ($all | Where-Object { $_.ProcessId -eq $r.Id } | Select-Object -First 1).ParentProcessId
        $where = if (-not $Root) { "" } elseif ($inTree.ContainsKey([int]$r.Id)) { " in-tree" } else { " OUTSIDE-tree" }
        Write-Host ("  {0} pid={1} parent={2} start={3:HH:mm:ss} cpu={4:N1}s{5}" -f $r.ProcessName, $r.Id, $parent, $r.StartTime, $r.CPU, $where)
    }
}

foreach ($p in @($Staged, $Packaged)) {
    if ($p -match '\s') { Fail "the BuildPlugin paths must not contain spaces (RunUAT.bat re-parses its arguments through cmd): $p" }
}
Show-BuildProcesses "already running before BuildPlugin"
Import-Module (Join-Path $PSScriptRoot "ci-log-stall-watch.psm1") -Force
$Uat = Invoke-StallWatched -FilePath (Join-Path $EngineRoot "Engine\Build\BatchFiles\RunUAT.bat") `
    -ArgumentList @("BuildPlugin", "-Plugin=$Staged\AliceSDF.uplugin", "-Package=$Packaged", "-TargetPlatforms=Win64", "-Rocket") `
    -LogPath $UatLog -StallSeconds ($StallMinutes * 60) -PollSeconds 15 `
    -OnHang { param($root) Show-BuildProcesses "at the hang, before the kill" $root }
if (-not $Uat.Drained) {
    Show-BuildProcesses "still holding BuildPlugin's output after it exited ($($Uat.DroppedLines) later line(s) dropped)"
}
# the PDB servers of this job's endpoint: started after $BuildStart (one job runs at a time on
# this runner); left running they would hold files of the next job's work directory
$OwnPdb = @(Get-Process -Name mspdbsrv -ErrorAction SilentlyContinue | Where-Object { $_.StartTime -ge $BuildStart })
foreach ($q in $OwnPdb) {
    Write-Host ("stopping this job's mspdbsrv pid={0} start={1:HH:mm:ss}" -f $q.Id, $q.StartTime)
    Stop-Process -Id $q.Id -Force -ErrorAction SilentlyContinue
}
$UatText = (Read-Shared $UatLog) + "`n" + (Read-Shared $Uat.StderrPath)
$UatText -split "`r?`n" | Where-Object { $_ -match "error|warning C|Result:|BUILD |\[\d+/\d+\]|executor" } | ForEach-Object { $_ }
if ($Uat.Hung) {
    Fail "HANG: BuildPlugin printed nothing for $StallMinutes min and was stopped; last line: $($Uat.LastLine)"
}
if ($Uat.ExitCode -ne 0) { Fail "BuildPlugin failed with exit code $($Uat.ExitCode) (see $UatLog)" }
# UBT writes its environment config under the writable engine directory, which for an installed
# engine is %LOCALAPPDATA%\UnrealEngine; it is rewritten from the environment on every run
$UbtEnvXml = Join-Path $env:LOCALAPPDATA "UnrealEngine\Intermediate\Build\UnrealBuildTool.Env.BuildConfiguration.xml"
$UbtEnvText = Read-Shared $UbtEnvXml
if ($null -eq $UbtEnvText) { Fail "UBT wrote no environment config at $UbtEnvXml" }
Write-Host "UBT environment config ($UbtEnvXml):"
Write-Host $UbtEnvText
if ($UbtEnvText -notmatch "<bAllowUBAExecutor>\s*false\s*</bAllowUBAExecutor>") {
    Fail "UnrealBuildTool_BuildConfiguration__bAllowUBAExecutor=false did not reach UBT (not in $UbtEnvXml)"
}
# UBT prints "Using <executor> executor to run N action(s)" for each target it compiles
if ($UatText -notmatch "Using .+ executor to run \d+ action") {
    Fail "BuildPlugin compiled nothing (no 'Using ... executor to run N action(s)' line, see $UatLog)"
}
if (-not $UatText.Contains("BUILD SUCCESSFUL")) {
    Fail "BuildPlugin exited 0 without BUILD SUCCESSFUL (see $UatLog)"
}
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
        @{ Name = "PythonScriptPlugin"; Enabled = $true },
        # Fab / Quixel Bridge open a CEF web view as the editor starts, and
        # CEF asserts on `BUseSupportedRHIRenderer()` under D3D12 in this
        # session (the runner reached FfiCorpusParity and then died there).
        # Nothing in these tests needs a browser.
        @{ Name = "Fab"; Enabled = $false },
        @{ Name = "Bridge"; Enabled = $false },
        @{ Name = "WebBrowserWidget"; Enabled = $false }
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
