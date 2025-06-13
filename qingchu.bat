@echo off
REM Step 1: 清除大文件
java -jar bfg.jar --delete-files "*torch_cpu.dll" --delete-files "*xgboost.dll" --delete-files "*_catboost.pyd" --delete-files "*dnnl.lib"

REM Step 2: 清理 Git 历史
git reflog expire --expire=now --all
git gc --prune=now --aggressive

REM Step 3: 强制推送清理后的代码
git push origin --force
