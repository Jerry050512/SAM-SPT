#!/bin/bash
set -e

# Get the absolute directory of the script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Load environment variables from .env file if it exists
if [ -f "${SCRIPT_DIR}/.env" ]; then
  # Export variables for use in sub-processes like the expect script
  export $(grep -v '^#' "${SCRIPT_DIR}/.env" | xargs)
  echo "Loaded credentials from .env file."
fi

# Perform OSS login using the expect script
# Note: 'expect' must be installed (e.g., 'sudo apt-get install expect')
LOGIN_SCRIPT="${SCRIPT_DIR}/oss_login.exp"
if [ -f "${LOGIN_SCRIPT}" ]; then
  echo "Attempting OSS login..."
  chmod +x "${LOGIN_SCRIPT}"
  "${LOGIN_SCRIPT}"
  echo "OSS login successful."
fi

python main.py
python main.py --eval --restore-model /hy-tmp/output/checkpoint.pth

cd /hy-tmp

# 压缩包名称
file="SAM-SPT-train-result-$(date "+%Y%m%d-%H%M%S").zip"
# 把 result 目录做成 zip 压缩包
zip -q -r "${file}" output

# 通过 oss 上传到个人数据中的 results 文件夹中
oss cp "${file}" oss://results/
rm -f "${file}"

# 传输成功后关机
shutdown