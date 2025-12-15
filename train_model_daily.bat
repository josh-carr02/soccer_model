@echo off
cd /d C:\Users\setti\Documents\soccer_model
call .\.venv\Scripts\activate.bat
python -m soccer_model.train_only
