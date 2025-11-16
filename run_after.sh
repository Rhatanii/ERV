#!/bin/bash

# 모니터링할 bash 파일 이름 (경로 포함 가능)
target_name="eval_shard-SFT_AU.sh"


# 실행이 끝난 후 실행할 Python 스크립트 경로
next_script="tools/response_gpt_check-my_own.sh"

echo "⏳ Waiting for '$target_name' to finish..."

# 프로세스가 존재할 동안 반복
while pgrep -f "$target_name" >/dev/null; do
    sleep 10
done

echo "✅ '$target_name' has finished. Running $next_script..."
bash "$next_script"

# bash "$target_name"

