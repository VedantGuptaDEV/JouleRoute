@echo off
rem Start the whole app on Windows from cmd: the three llama-servers, the gateway, then the chat;
rem everything this started is stopped when the chat exits. See scripts\run_app.py.
rem Arguments go to cli\joule.py, e.g. scripts\run_app.cmd -p "what is 17 * 23"
setlocal
cd /d "%~dp0.."
if not exist .venv\Scripts\python.exe (echo .venv not found: run scripts\setup.cmd first & exit /b 1)
.venv\Scripts\python.exe scripts\run_app.py %*
