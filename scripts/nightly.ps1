# Nightly job wrapper (Windows). Logs to store\logs\nightly_YYYYMMDD.log
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force "$root\store\logs" | Out-Null
$log = "$root\store\logs\nightly_$(Get-Date -Format yyyyMMdd).log"
& "$root\.venv\Scripts\python.exe" -m fintailor nightly *>> $log
