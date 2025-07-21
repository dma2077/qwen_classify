# 1. 暂停历史记录，防止删除操作本身被记录
set +o history

history -d 
history -w

# 4. 恢复历史记录
set -o history
