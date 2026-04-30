cd /root/cce || exit 1

for c in car cyberpunk dog glasses red van_gogh; do
  python -m runtime.shared.batch \
    --prompts_path /root/cce/batch_test_prompt/${c}.csv \
    --concepts ${c}
done
