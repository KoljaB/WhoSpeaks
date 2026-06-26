param(
    [int]$Port = 8767,
    [string]$TraceLog = "",
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$env:WHO_SPEAKS_ROOT = $repoRoot
$env:REALTIMESTT_ROOT = "D:\Projekte\STT\RealtimeSTT\RealtimeSTT"
$env:PYTHONPATH = "$repoRoot;$env:REALTIMESTT_ROOT"

$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$embeddingPython = Join-Path $repoRoot ".venv-voice-embeddings\Scripts\python.exe"

$arguments = @(
    "tools\realtime_speakerdiarize.py",
    "--embedding-python",
    $embeddingPython,
    "--port",
    "$Port"
)

if ($NoBrowser) {
    $arguments += "--no-browser"
}

if ($TraceLog) {
    $arguments += @("--trace-log", $TraceLog)
}

& $pythonExe @arguments
