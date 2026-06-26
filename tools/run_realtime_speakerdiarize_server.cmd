@echo off
setlocal

cd /d "%~dp0\.." || exit /b 1

set "PYTHONPATH=%CD%;D:\Projekte\STT\RealtimeSTT\RealtimeSTT"

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8767"

set "TRACE_LOG=%~2"
if "%TRACE_LOG%"=="" set "TRACE_LOG=tools\.realtime_speakerdiarize_traces\trace-cmd-server.jsonl"

set "SERVER_LOG=tools\.realtime_speakerdiarize_traces\trace-cmd-server.process.log"

".\.venv\Scripts\python.exe" tools\realtime_speakerdiarize.py ^
  --embedding-python ".\.venv-voice-embeddings\Scripts\python.exe" ^
  --no-browser ^
  --port "%PORT%" ^
  --trace-log "%TRACE_LOG%" ^
  > "%SERVER_LOG%" 2>&1

exit /b %ERRORLEVEL%
