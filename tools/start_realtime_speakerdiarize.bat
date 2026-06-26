@echo off
setlocal

pushd "%~dp0\.." || (
    echo Failed to change to the WhoSpeaks folder.
    pause
    exit /b 1
)

set "WHO_SPEAKS_ROOT=%CD%"
set "REALTIMESTT_ROOT=D:\Projekte\STT\RealtimeSTT\RealtimeSTT"
set "PYTHON_EXE=%WHO_SPEAKS_ROOT%\.venv\Scripts\python.exe"
set "EMBEDDING_PYTHON=%WHO_SPEAKS_ROOT%\.venv-voice-embeddings\Scripts\python.exe"
set "PYTHONPATH=%WHO_SPEAKS_ROOT%;%REALTIMESTT_ROOT%"

if not exist "%PYTHON_EXE%" (
    echo Python executable not found:
    echo   %PYTHON_EXE%
    popd
    pause
    exit /b 1
)

if not exist "%EMBEDDING_PYTHON%" (
    echo Embedding Python executable not found:
    echo   %EMBEDDING_PYTHON%
    popd
    pause
    exit /b 1
)

"%PYTHON_EXE%" tools\realtime_speakerdiarize.py --embedding-python "%EMBEDDING_PYTHON%" %*
set "WHO_SPEAKS_EXIT_CODE=%ERRORLEVEL%"

popd

if not "%WHO_SPEAKS_EXIT_CODE%"=="0" (
    echo.
    echo realtime_speakerdiarize.py exited with error code %WHO_SPEAKS_EXIT_CODE%.
    pause
)

exit /b %WHO_SPEAKS_EXIT_CODE%
