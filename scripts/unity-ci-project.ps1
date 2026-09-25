# Creates the throw-away VRChat Worlds project the Unity CI job runs the checks
# in, on Windows (the self-hosted runner). Same contract as
# scripts/unity-ci-project.sh (the Linux / container variant): the project
# skeleton comes from vrchat-package/CI~, com.vrchat.worlds is resolved with
# vrc-get, and com.alice.sdf is linked to this checkout as a file: dependency.
#
#   pwsh scripts/unity-ci-project.ps1 [-Project ci-unity]
#
# Idempotent: an existing Library/ is kept, which is most of a run's time on a
# self-hosted runner. The SDK version is pinned here so a VRChat release cannot
# turn CI red on its own; bump it in both this file and unity-ci-project.sh.
[CmdletBinding()]
param(
    [string]$Project = "ci-unity",
    [string]$SdkVersion = "3.10.5",
    [string]$VrcGetVersion = "1.9.2"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$proj = Join-Path $root $Project
New-Item -ItemType Directory -Force (Join-Path $proj "Assets") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $proj "Packages") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $proj "ProjectSettings") | Out-Null
Copy-Item (Join-Path $root "vrchat-package\CI~\ProjectVersion.txt") (Join-Path $proj "ProjectSettings\ProjectVersion.txt") -Force
# the template's file: link is ../../vrchat-package relative to Packages/, so the
# project must sit directly under the repository root
Copy-Item (Join-Path $root "vrchat-package\CI~\manifest.json") (Join-Path $proj "Packages\manifest.json") -Force

$tools = Join-Path $root "target\ci-tools"
New-Item -ItemType Directory -Force $tools | Out-Null
$vrcGet = Join-Path $tools "vrc-get.exe"
if (-not (Test-Path $vrcGet)) {
    $url = "https://github.com/vrc-get/vrc-get/releases/download/v$VrcGetVersion/x86_64-pc-windows-msvc-vrc-get.exe"
    Write-Host "unity-ci-project: downloading vrc-get $VrcGetVersion"
    Invoke-WebRequest -Uri $url -OutFile $vrcGet -UseBasicParsing
}

& $vrcGet install com.vrchat.worlds $SdkVersion --project $proj --yes
if ($LASTEXITCODE -ne 0) { Write-Host "unity-ci-project: vrc-get failed ($LASTEXITCODE)"; exit 1 }
foreach ($pkg in @("com.vrchat.worlds", "com.vrchat.base")) {
    if (-not (Test-Path (Join-Path $proj "Packages\$pkg"))) { Write-Host "unity-ci-project: $pkg not installed"; exit 1 }
}
Write-Host "unity-ci-project: $proj ready (com.vrchat.worlds $SdkVersion, com.alice.sdf -> vrchat-package)"
