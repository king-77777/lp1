# 设置项目目录（可手动修改）
cd "D:\study\ren gong zhi neng\RZZS"

# 检查是否有 BFG 工具
if (!(Test-Path ".\bfg.jar")) {
    Write-Host "❌ 请先下载 bfg.jar 到当前目录：https://rtyley.github.io/bfg-repo-cleaner/" -ForegroundColor Red
    exit
}

# 第一步：忽略大文件
Set-Content .gitignore @"
venv/
env/
__pycache__/
*.py[cod]
*.dll
*.pyd
*.lib
*.pkl
*.h5
*.joblib
*.log
*.csv
*.tsv
*.xlsx
*.xls
.ipynb_checkpoints/
.DS_Store
Thumbs.db
"@

# 第二步：清除历史中的大文件
java -jar .\bfg.jar --delete-files "*torch_cpu.dll" --delete-files "*xgboost.dll" --delete-files "*_catboost.pyd" --delete-files "*dnnl.lib"

# 第三步：彻底清理 Git 垃圾
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 第四步：再次添加和提交（防止忽略文件已跟踪）
git add .gitignore
git rm -r --cached venv
git commit -am "清理大文件，添加.gitignore"

# 第五步：强制推送
git push origin main --force
