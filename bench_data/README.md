# LongMemEval 数据（体积大，用 fetch_benchmarks.py 下载）
# 本文件存在即代表该目录在仓库里，但 *.json / *.pkl 不入库。
#
#   python fetch_benchmarks.py                 # 下载 oracle + s
#   python fetch_benchmarks.py --include m     # 连 500 会话的 m 版
#   python fetch_benchmarks.py --repo locomo   # 下载 LoCoMo
#
# 说明：HuggingFace 官方域名在部分网络环境会超时，脚本已默认走 hf-mirror.com 镜像。
