#!/bin/bash

# 遍历当前目录下的所有项，带 / 结尾代表只匹配目录
for dir in */; do
    # 确保当前遍历到的是一个真正的目录
    if [ -d "$dir" ]; then
        echo "========================================"
        echo "正在处理目录: $dir"
        
        # 进入子目录，如果进入失败（例如权限问题），则跳过当前目录
        cd "$dir" || continue
        
        # 执行 sbatch 提交任务
        rm -rf *.out *.err
        sbatch ../../../stab/train_ddg.sh
        
        # 返回上一级（也就是原先的起点目录）
        cd ..
        
        # 停顿5秒
        echo "等待 5 秒..."
        sleep 5
    fi
done

echo "========================================"
echo "所有任务提交完毕！"
