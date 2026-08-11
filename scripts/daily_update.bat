@echo off
REM ============================================================
REM  Daily job (runs 17:00): fetch new items -> re-index -> re-cluster
REM  Triggered by Windows Task Scheduler task "HeraklionRAG-DailyUpdate".
REM ============================================================
setlocal
set "PY=C:\Users\User\anaconda3\envs\rag\python.exe"
set "PROJ=C:\Users\User\Desktop\Vitsakis\info_retrieval"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not exist "%PROJ%\scripts\logs" mkdir "%PROJ%\scripts\logs"
set "LOG=%PROJ%\scripts\logs\daily.log"

echo. >> "%LOG%"
echo ==================== DAILY UPDATE  %date% %time% ==================== >> "%LOG%"

REM 1+2) crawl only new items per category AND rebuild the search index
"%PY%" "%PROJ%\crawler\daily_update.py" --rebuild-index >> "%LOG%" 2>&1

REM 3) rebuild the passage (chunk) index
"%PY%" "%PROJ%\indexing\build_chunks.py" >> "%LOG%" 2>&1

echo ==================== DONE  %date% %time% ==================== >> "%LOG%"
endlocal
