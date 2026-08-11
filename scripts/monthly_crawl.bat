@echo off
REM ============================================================
REM  Monthly job (runs on the 1st, 02:00): full crawl -> index -> cluster
REM  Triggered by Windows Task Scheduler task "HeraklionRAG-MonthlyCrawl".
REM  This is the completeness backstop; the daily job keeps things current.
REM ============================================================
setlocal
set "PY=C:\Users\User\anaconda3\envs\rag\python.exe"
set "PROJ=C:\Users\User\Desktop\Vitsakis\info_retrieval"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not exist "%PROJ%\scripts\logs" mkdir "%PROJ%\scripts\logs"
set "LOG=%PROJ%\scripts\logs\monthly.log"

echo. >> "%LOG%"
echo ==================== MONTHLY FULL CRAWL  %date% %time% ==================== >> "%LOG%"

REM 1) full crawl (incremental via manifest; may run many hours)
"%PY%" "%PROJ%\crawler\crawler.py" >> "%LOG%" 2>&1

REM 2) rebuild the search index
"%PY%" "%PROJ%\indexing\build_index.py" >> "%LOG%" 2>&1

REM 3) rebuild the passage (chunk) index
"%PY%" "%PROJ%\indexing\build_chunks.py" >> "%LOG%" 2>&1

echo ==================== DONE  %date% %time% ==================== >> "%LOG%"
endlocal
