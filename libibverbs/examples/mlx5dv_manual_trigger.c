

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/time.h>
#include <endian.h>
#include <errno.h>
#include "log.h"
#include "../verbs.h"
#include <infiniband/mlx5dv.h>

#define DEFAULT_MSG_SIZE 64
#define MAX_POLL_CQ_TIMEOUT 5000

/* MLX5 CQ doorbell constants */
#define MLX5_CQ_SET_CI 0

/* 系统资源结构体 */
struct resources
{
	struct ibv_device_attr device_attr;  /* 设备属性 */
	struct ibv_port_attr port_attr;      /* IB端口属性 */
	struct ibv_context *ib_ctx;          /* 设备句柄 */
	struct ibv_pd *pd;                   /* PD句柄 */
	struct ibv_cq *cq;                   /* CQ句柄 */
	struct ibv_qp *qp;                   /* QP句柄 */
	struct ibv_mr *mr;                   /* MR句柄 */
	char *buf;                           /* 内存缓冲区指针 */
	uint32_t max_inline_data;            /* 最大inline数据大小 */
	struct mlx5dv_qp mlx5dv_qp;          /* QP信息（标准API，用于手动构建WQE） */
	struct mlx5dv_cq mlx5dv_cq;          /* CQ信息（标准API，用于手动poll CQ） */
	uint32_t sq_cur_post;                /* 当前SQ producer index */
	uint32_t cq_cons_index;              /* 当前CQ consumer index */
	uint32_t bf_offset;                  /* Blueflame offset（自行维护，初始值为0） */
};

/* 配置结构体 */
struct config_t
{
	const char *dev_name;  /* IB设备名称 */
	int ib_port;          /* 本地IB端口 */
	int gid_idx;          /* GID索引 */
	size_t msg_size;      /* 消息缓冲区大小 */
	int repeat_count;     /* 重复测试次数 */
};

static struct config_t config = {
	NULL,      /* dev_name */
	1,         /* ib_port */
	-1,        /* gid_idx */
	DEFAULT_MSG_SIZE,  /* msg_size */
	1          /* repeat_count */
};

/*
 * 手动读取CQE（用于GPU核函数）
 * 
 * ========== CQ (Completion Queue) 工作原理 ==========
 * 
 * 1. CQ 基本结构：
 *    - CQ 是一个环形缓冲区，用于存储完成队列条目（CQE）
 *    - 硬件（HCA）作为生产者：完成操作后写入 CQE
 *    - 软件（CPU/GPU）作为消费者：轮询并读取 CQE
 * 
 * 2. 环形缓冲区机制：
 *    - CQ 有 cqe_cnt 个 CQE 位置（通常是 2 的幂，如 16、32、64）
 *    - 使用取模运算实现环形：position = index % cqe_cnt
 *    - 当写满后，硬件会循环回到第一个位置重用
 * 
 * 3. Ownership Bit（所有权位）机制：
 *    - 这是 CQ 的核心同步机制，实现硬件和软件的无锁同步
 *    - Owner Bit 含义：
 *      * owner = expected_owner：硬件刚写入的新 CQE，软件可以读取
 *      * owner ≠ expected_owner：旧 CQE（软件已读）或无效位置，软件应跳过
 * 
 * 4. Polarity（极性）机制：
 *    - Expected Owner 的计算：expected = !!(consumer_index & cqe_cnt)
 *    - 每 cqe_cnt 个 CQE，Expected Owner 翻转一次（0→1→0→1...）
 *    - 硬件写入时：根据 Producer Index 计算 owner = !!(producer_index & cqe_cnt)
 *    - 软件检查时：根据 Consumer Index 计算 expected = !!(consumer_index & cqe_cnt)
 *    - 当 producer_index >= consumer_index 且在同一 polarity 区间时，owner 会匹配
 * 
 *    示例（cqe_cnt = 4）：
 *    Consumer Index | Expected Owner | 说明
 *    ─────────────────────────────────────────────
 *    0-3            | 0              | 第一轮，期望 owner=0
 *    4-7            | 1              | 第二轮，期望 owner=1
 *    8-11           | 0              | 第三轮，期望 owner=0
 *    ...
 * 
 * 5. Consumer Index 和 Doorbell：
 *    - Consumer Index：软件已读取的 CQE 数量（单调递增）
 *    - Doorbell：软件通过 doorbell 通知硬件当前的 Consumer Index
 *    - 硬件收到 doorbell 后，知道可以重用 consumer_index 之前的位置
 *    - 更新 doorbell：dbrec[MLX5_CQ_SET_CI] = consumer_index
 * 
 * 6. 工作流程：
 *    (1) 硬件完成操作 → 写入 CQE 到环形缓冲区
 *        └─> 根据 Producer Index 计算 owner = !!(producer_index & cqe_cnt)
 *        └─> 设置 CQE 的 owner bit
 *    
 *    (2) 软件轮询 CQ：
 *        ├─> 计算 CQE 地址（环形索引）
 *        ├─> 检查 opcode != INVALID
 *        ├─> 检查 ownership bit == expected_owner
 *        ├─> 内存屏障（确保读取顺序）
 *        ├─> 读取 CQE 内容并解析
 *        ├─> consumer_index++
 *        └─> 更新 doorbell（通知硬件）
 *    
 *    (3) 硬件收到 doorbell：
 *        └─> 知道软件已消费，可以重用该 CQE 位置
 * 
 * 7. 为什么需要这个机制？
 *    - 无锁同步：通过 ownership bit 和 polarity 实现硬件和软件的无锁同步
 *    - 高效：环形缓冲区避免频繁的内存分配/释放
 *    - 硬件友好：doorbell 机制减少中断开销
 *    - 可扩展：支持 64/128 字节 CQE，适应不同场景
 * 
 * ========== 完整示例：逐步演示 CQ 工作原理 ==========
 * 
 * 假设：cqe_cnt = 4（4 个 CQE 位置），初始状态 Consumer Index = 0
 * 
 * 【第一轮：硬件写入 CQE0, CQE1, CQE2, CQE3】
 * 
 * 硬件写入第 0 个 CQE（CQE0）：
 *   - Producer Index = 0
 *   - Position = 0 % 4 = 0 (CQE0)
 *   - Owner = !!(0 & 4) = 0
 *   - 硬件设置：CQE0.owner = 0
 * 
 * 硬件写入第 1 个 CQE（CQE1）：
 *   - Producer Index = 1
 *   - Position = 1 % 4 = 1 (CQE1)
 *   - Owner = !!(1 & 4) = 0
 *   - 硬件设置：CQE1.owner = 0
 * 
 * 硬件写入第 2 个 CQE（CQE2）：
 *   - Producer Index = 2
 *   - Position = 2 % 4 = 2 (CQE2)
 *   - Owner = !!(2 & 4) = 0
 *   - 硬件设置：CQE2.owner = 0
 * 
 * 硬件写入第 3 个 CQE（CQE3）：
 *   - Producer Index = 3
 *   - Position = 3 % 4 = 3 (CQE3)
 *   - Owner = !!(3 & 4) = 0
 *   - 硬件设置：CQE3.owner = 0
 * 
 * 此时状态：
 *   CQ 缓冲区：
 *   ┌──────┬──────┬──────┬──────┐
 *   │ CQE0 │ CQE1 │ CQE2 │ CQE3 │
 *   │owner=0│owner=0│owner=0│owner=0│
 *   └──────┴──────┴──────┴──────┘
 *   Consumer Index = 0（软件还没读）
 *   Producer Index = 4（硬件已写4个）
 * 
 * 【软件读取 CQE0】
 * 
 * 软件轮询：
 *   - Consumer Index = 0
 *   - Expected Owner = !!(0 & 4) = 0
 *   - 计算位置：0 % 4 = 0 (CQE0)
 *   - 读取 CQE0.owner = 0
 *   - 检查：owner(0) == expected(0) ✓ 匹配！
 *   - 软件读取 CQE0 内容
 *   - Consumer Index++ → 1
 *   - 软件更新 doorbell：Consumer Index = 1（通知硬件）
 * 
 * 此时状态：
 *   CQ 缓冲区：
 *   ┌──────┬──────┬──────┬──────┐
 *   │ CQE0 │ CQE1 │ CQE2 │ CQE3 │
 *   │owner=0│owner=0│owner=0│owner=0│  ← CQE0 软件已读
 *   └──────┴──────┴──────┴──────┘
 *   Consumer Index = 1（软件已读1个）
 *   Producer Index = 4（硬件已写4个）
 *   Doorbell = 1（硬件知道软件已读1个）
 * 
 * 硬件要写入第 CQE4（第二轮开始，重用 CQE0）
 * 
 * 硬件写入 CQE4：
 *   - Producer Index = 4
 *   - Position = 4 % 4 = 0 (CQE0)
 *   - Owner = !!(4 & 4) = 1  ← 关键！第二轮 owner 变成 1
 *   - 硬件设置：CQE0.owner = 1（覆盖旧数据）
 * 
 * 此时状态：
 *   CQ 缓冲区：
 *   ┌──────┬──────┬──────┬──────┐
 *   │ CQE0 │ CQE1 │ CQE2 │ CQE3 │
 *   │owner=1│owner=0│owner=0│owner=0│  ← CQE0 被重用，owner=1
 *   └──────┴──────┴──────┴──────┘
 *   Consumer Index = 1（软件已读1个）
 *   Producer Index = 5（硬件已写5个）
 * 
 * 【软件继续读取 CQE1, CQE2, CQE3】
 * 
 * 软件读取 CQE1：
 *   - Consumer Index = 1
 *   - Expected Owner = !!(1 & 4) = 0
 *   - CQE1.owner = 0
 *   - 检查：owner(0) == expected(0) ✓ 匹配！
 *   - Consumer Index++ → 2
 * 
 * 软件读取 CQE2：
 *   - Consumer Index = 2
 *   - Expected Owner = !!(2 & 4) = 0
 *   - CQE2.owner = 0
 *   - 检查：owner(0) == expected(0) ✓ 匹配！
 *   - Consumer Index++ → 3
 * 
 * 软件读取 CQE3：
 *   - Consumer Index = 3
 *   - Expected Owner = !!(3 & 4) = 0
 *   - CQE3.owner = 0
 *   - 检查：owner(0) == expected(0) ✓ 匹配！
 *   - Consumer Index++ → 4
 * 
 * 【软件读取第二轮的第一个 CQE（CQE0）】
 * 
 * 软件轮询 CQE0：
 *   - Consumer Index = 4
 *   - Expected Owner = !!(4 & 4) = 1  ← 关键！第二轮 expected 变成 1
 *   - 计算位置：4 % 4 = 0 (CQE0)
 *   - 读取 CQE0.owner = 1
 *   - 检查：owner(1) == expected(1) ✓ 匹配！
 *   - 软件读取 CQE0 内容（第二轮的新数据）
 *   - Consumer Index++ → 5
 * 
 * 此时状态：
 *   CQ 缓冲区：
 *   ┌──────┬──────┬──────┬──────┐
 *   │ CQE0 │ CQE1 │ CQE2 │ CQE3 │
 *   │owner=1│owner=0│owner=0│owner=0│  ← CQE0 软件已读（第二轮）
 *   └──────┴──────┴──────┴──────┘
 *   Consumer Index = 5（软件已读5个）
 *   Producer Index = 5（硬件已写5个）
 * 
 * 硬件要写入 CQE5：
 *   - Producer Index = 5
 *   - Position = 5 % 4 = 1 (CQE1)
 *   - Owner = !!(5 & 4) = 1  ← 第二轮 owner = 1
 *   - 硬件设置：CQE1.owner = 1
 * 
 * 软件检查 CQE1：
 *   - Consumer Index = 5
 *   - Expected Owner = !!(5 & 4) = 1
 *   - CQE1.owner = 1
 *   - 检查：owner(1) == expected(1) ✓ 匹配！
 * 
 * ========== 关键理解点 ==========
 * 
 * 1. 硬件写入时：
 *    - 根据 Producer Index 计算：owner = !!(producer_index & cqe_cnt)
 *    - 不是根据 Consumer Index 计算
 * 
 * 2. 软件检查时：
 *    - 根据 Consumer Index 计算：expected = !!(consumer_index & cqe_cnt)
 *    - 当 producer_index >= consumer_index 且在同一 polarity 区间时匹配
 * 
 * 3. Polarity 翻转：
 *    - 每 cqe_cnt 个 CQE，Expected Owner 翻转一次
 *    - 第一轮（0-3）：expected = 0，硬件写 owner = 0
 *    - 第二轮（4-7）：expected = 1，硬件写 owner = 1
 *    - 第三轮（8-11）：expected = 0，硬件写 owner = 0
 *    - ...
 * 
 * 4. 为什么这样设计？
 *    - 即使硬件覆盖了旧 CQE，软件也能通过 owner 是否匹配 expected 来判断
 *    - 匹配 → 新 CQE，可以读取
 *    - 不匹配 → 旧 CQE 或无效，跳过
 */
static int manual_poll_cq(struct resources *res, struct ibv_wc *wc)
{
	void *cqe;
	struct mlx5_cqe64 *cqe64;
	uint8_t opcode;
	uint32_t cqe_idx;
	
	/* 
	 * 步骤1：计算当前要读取的 CQE 地址
	 * - 使用环形索引：cqe_idx = cons_index & (cqe_cnt - 1)
	 * - 由于 cqe_cnt 是 2 的幂，& 操作等价于 % 操作，但更高效
	 * - 地址 = 基地址 + 索引 × CQE 大小
	 */
	cqe_idx = res->cq_cons_index & (res->mlx5dv_cq.cqe_cnt - 1);
	cqe = (char *)res->mlx5dv_cq.buf + (cqe_idx * res->mlx5dv_cq.cqe_size);
	
	/* 
	 * 步骤2：获取 CQE64 结构体指针
	 * - 64 字节 CQE：CQE64 从偏移 0 开始
	 * - 128 字节 CQE：前 64 字节为压缩/扩展信息，CQE64 从偏移 64 开始
	 */
	cqe64 = (res->mlx5dv_cq.cqe_size == 64) ? (struct mlx5_cqe64 *)cqe : 
		(struct mlx5_cqe64 *)((char *)cqe + 64);
	
	/* 
	 * 步骤3：检查 opcode 是否为 INVALID
	 * - MLX5_CQE_INVALID (15) 表示无效/空的 CQE
	 * - 这是快速路径检查，可以提前判断 CQ 是否为空
	 */
	opcode = mlx5dv_get_cqe_opcode(cqe64);
	if (opcode == MLX5_CQE_INVALID) {
		return 0;  /* CQ为空 */
	}
	
	/* 
	 * 步骤4：检查 Ownership Bit（核心同步机制）
	 * - owner：CQE 中的 ownership bit（硬件写入时设置）
	 * - expected_owner：根据 Consumer Index 计算的期望值
	 *   * 公式：!!(consumer_index & cqe_cnt)
	 *   * 每 cqe_cnt 个 CQE，expected_owner 翻转一次
	 * - 匹配条件：
	 *   * owner == expected_owner：硬件新写入的 CQE，可以读取
	 *   * owner != expected_owner：旧 CQE 或无效位置，跳过
	 * 
	 * 工作原理：
	 * - 硬件写入第 n 个 CQE 时：owner = !!(n & cqe_cnt)
	 * - 软件检查时：expected = !!(consumer_index & cqe_cnt)
	 * - 当 producer_index >= consumer_index 且在同一 polarity 区间时匹配
	 */
	uint8_t owner = mlx5dv_get_cqe_owner(cqe64);
	uint8_t expected_owner = !!(res->cq_cons_index & res->mlx5dv_cq.cqe_cnt);
	if (owner != expected_owner) {
		return 0;  /* CQ为空或这是旧数据 */
	}
	
	/* 
	 * 步骤5：内存屏障（确保读取顺序）
	 * - 先检查 ownership bit 和 opcode（快速路径）
	 * - 确认有效后再读取 CQE 内容
	 * - 内存屏障确保不会乱序读取到未完成的数据
	 */
	__sync_synchronize();
	
	/* 
	 * 步骤6：解析 CQE 内容
	 * - 初始化 work completion 结构
	 * - 提取 QP 号（低 24 位）
	 * - 提取传输的字节数
	 */
	memset(wc, 0, sizeof(*wc));
	wc->qp_num = be32toh(cqe64->sop_drop_qpn) & 0xffffff;  /* QP 号（低24位） */
	wc->byte_len = be32toh(cqe64->byte_cnt);              /* 传输的字节数 */
	
	/* 
	 * 步骤7：根据 opcode 设置 work completion 的操作类型
	 * CQE Opcode 分类：
	 * - MLX5_CQE_REQ (0)：请求完成（Send/RDMA Write/RDMA Read 等）
	 * - MLX5_CQE_RESP_SEND (2)：响应完成（Recv）
	 * - MLX5_CQE_RESP_SEND_IMM (3)：带立即数的响应
	 * - MLX5_CQE_RESP_WR_IMM (1)：RDMA Write 带立即数
	 * - MLX5_CQE_REQ_ERR (13)：请求错误
	 * - MLX5_CQE_RESP_ERR (14)：响应错误
	 */
	switch (opcode) {
	case MLX5_CQE_REQ:
		/* 
		 * 请求完成（Send/RDMA Write 等）
		 * - sop_drop_qpn 字段的高 8 位（bit 24-31）存储操作码
		 * - 需要根据具体操作码设置对应的 IBV_WC_* 类型
		 */
		switch (be32toh(cqe64->sop_drop_qpn) >> 24) {
		case MLX5_OPCODE_RDMA_WRITE:
		case MLX5_OPCODE_RDMA_WRITE_IMM:
			wc->opcode = IBV_WC_RDMA_WRITE;
			if ((be32toh(cqe64->sop_drop_qpn) >> 24) == MLX5_OPCODE_RDMA_WRITE_IMM)
				wc->wc_flags |= IBV_WC_WITH_IMM;
			break;
		case MLX5_OPCODE_SEND:
		case MLX5_OPCODE_SEND_IMM:
		case MLX5_OPCODE_SEND_INVAL:
			wc->opcode = IBV_WC_SEND;
			if ((be32toh(cqe64->sop_drop_qpn) >> 24) == MLX5_OPCODE_SEND_IMM)
				wc->wc_flags |= IBV_WC_WITH_IMM;
			break;
		case MLX5_OPCODE_RDMA_READ:
			wc->opcode = IBV_WC_RDMA_READ;
			break;
		default:
			wc->opcode = IBV_WC_SEND;  /* 默认 */
			break;
		}
		wc->status = IBV_WC_SUCCESS;
		break;
	case MLX5_CQE_RESP_SEND:
		wc->opcode = IBV_WC_RECV;
		wc->status = IBV_WC_SUCCESS;
		break;
	case MLX5_CQE_RESP_SEND_IMM:
		wc->opcode = IBV_WC_RECV;
		wc->wc_flags |= IBV_WC_WITH_IMM;
		wc->imm_data = be32toh(cqe64->imm_inval_pkey);
		wc->status = IBV_WC_SUCCESS;
		break;
	case MLX5_CQE_RESP_WR_IMM:
		wc->opcode = IBV_WC_RECV_RDMA_WITH_IMM;
		wc->wc_flags |= IBV_WC_WITH_IMM;
		wc->imm_data = be32toh(cqe64->imm_inval_pkey);
		wc->status = IBV_WC_SUCCESS;
		break;
	case MLX5_CQE_REQ_ERR:
	case MLX5_CQE_RESP_ERR:
		/* 错误处理 */
		wc->status = IBV_WC_GENERAL_ERR;
		break;
	default:
		return 0;  /* 未知opcode，跳过 */
	}
	
	/* 
	 * 步骤8：更新 Consumer Index
	 * - Consumer Index 表示软件已读取的 CQE 数量（单调递增）
	 * - 每次成功读取一个 CQE 后递增
	 */
	res->cq_cons_index++;
	
	/* 
	 * 步骤9：更新 Doorbell（通知硬件）
	 * - Doorbell 是硬件和软件之间的同步机制
	 * - 软件通过 doorbell 通知硬件当前的 Consumer Index
	 * - 硬件收到后，知道可以重用 consumer_index 之前的位置
	 * - & 0xffffff：只取低 24 位（硬件限制）
	 * - htobe32：转换为大端字节序（网络字节序）
	 */
	res->mlx5dv_cq.dbrec[MLX5_CQ_SET_CI] = htobe32(res->cq_cons_index & 0xffffff);
	
	return 1;  /* 成功读取一个CQE */
}

/* 轮询完成队列 */
static int poll_completion(struct resources *res)
{
	struct ibv_wc wc;
	unsigned long start_time_msec;
	unsigned long cur_time_msec;
	struct timeval cur_time;
	int poll_result;
	int rc = 0;
	
	gettimeofday(&cur_time, NULL);
	start_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
	do
	{
		/* 使用手动poll CQ而不是ibv_poll_cq */
		poll_result = manual_poll_cq(res, &wc);
		gettimeofday(&cur_time, NULL);
		cur_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
	} while ((poll_result == 0) && ((cur_time_msec - start_time_msec) < MAX_POLL_CQ_TIMEOUT));
	
	if (poll_result < 0)
	{
		loge("poll CQ failed");
		rc = 1;
	}
	else if (poll_result == 0)
	{
		loge("completion wasn't found in the CQ after timeout");
		rc = 1;
	}
	else
	{
		const char *opcode_str;
		switch (wc.opcode) {
		case IBV_WC_SEND:
			opcode_str = "SEND";
			break;
		case IBV_WC_RDMA_WRITE:
			opcode_str = "RDMA_WRITE";
			break;
		case IBV_WC_RDMA_READ:
			opcode_str = "RDMA_READ";
			break;
		case IBV_WC_RECV:
			opcode_str = "RECV";
			break;
		case IBV_WC_RECV_RDMA_WITH_IMM:
			opcode_str = "RECV_RDMA_WITH_IMM";
			break;
		default:
			opcode_str = "UNKNOWN";
			break;
		}
		logi("completion was found in CQ with status 0x%x, opcode=%d (%s)", 
				wc.status, wc.opcode, opcode_str);
		if (wc.status != IBV_WC_SUCCESS)
		{
			loge("got bad completion with status: 0x%x, vendor syndrome: 0x%x", 
					wc.status, wc.vendor_err);
			rc = 1;
		}
	}
	return rc;
}

/* 初始化资源结构 */
static void resources_init(struct resources *res)
{
	memset(res, 0, sizeof(*res));
}

/* 创建RDMA资源 */
static int resources_create(struct resources *res)
{
	struct ibv_device **dev_list = NULL;
	struct ibv_qp_init_attr qp_init_attr;
	struct ibv_device *ib_dev = NULL;
	size_t size;
	int i;
	int mr_flags = 0;
	int num_devices;
	int rc = 0;
	
	logi("searching for IB devices in host");
	dev_list = ibv_get_device_list(&num_devices);
	if (!dev_list)
	{
		loge("failed to get IB devices list");
		rc = 1;
		goto resources_create_exit;
	}
	
	if (!num_devices)
	{
		loge("found %d device(s)", num_devices);
		rc = 1;
		goto resources_create_exit;
	}
	
	logi("found %d device(s)", num_devices);
	
	/* 查找指定的设备 */
	for (i = 0; i < num_devices; i++)
	{
		if (!config.dev_name)
		{
			config.dev_name = strdup(ibv_get_device_name(dev_list[i]));
			logi("device not specified, using first one found: %s", config.dev_name);
		}
		if (!strcmp(ibv_get_device_name(dev_list[i]), config.dev_name))
		{
			ib_dev = dev_list[i];
			break;
		}
	}
	
	if (!ib_dev)
	{
		loge("IB device %s wasn't found", config.dev_name);
		rc = 1;
		goto resources_create_exit;
	}
	
	/* 打开设备 */
	res->ib_ctx = ibv_open_device(ib_dev);
	if (!res->ib_ctx)
	{
		loge("failed to open device %s", config.dev_name);
		rc = 1;
		goto resources_create_exit;
	}
	
	ibv_free_device_list(dev_list);
	dev_list = NULL;
	ib_dev = NULL;
	
	/* 查询端口属性 */
	if (ibv_query_port(res->ib_ctx, config.ib_port, &res->port_attr))
	{
		loge("ibv_query_port on port %u failed", config.ib_port);
		rc = 1;
		goto resources_create_exit;
	}
	
	/* 分配保护域 */
	res->pd = ibv_alloc_pd(res->ib_ctx);
	if (!res->pd)
	{
		loge("ibv_alloc_pd failed");
		rc = 1;
		goto resources_create_exit;
	}
	
	/* 创建完成队列 */
	res->cq = ibv_create_cq(res->ib_ctx, 1, NULL, NULL, 0);
	if (!res->cq)
	{
		loge("failed to create CQ");
		rc = 1;
		goto resources_create_exit;
	}
	
	/* 分配内存缓冲区 */
	size = config.msg_size;
	res->buf = (char *)malloc(size);
	if (!res->buf)
	{
		loge("failed to malloc %zu bytes to memory buffer", size);
		rc = 1;
		goto resources_create_exit;
	}
	memset(res->buf, 0, size);
	
	/* 注册内存区域 */
	mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
	res->mr = ibv_reg_mr(res->pd, res->buf, size, mr_flags);
	if (!res->mr)
	{
		loge("ibv_reg_mr failed with mr_flags=0x%x", mr_flags);
		rc = 1;
		goto resources_create_exit;
	}
	logi("MR was registered with addr=%p, lkey=0x%x, rkey=0x%x", 
			res->buf, res->mr->lkey, res->mr->rkey);
	
	/* 创建队列对 */
	memset(&qp_init_attr, 0, sizeof(qp_init_attr));
	qp_init_attr.qp_type = IBV_QPT_RC;
	qp_init_attr.sq_sig_all = 1;
	qp_init_attr.send_cq = res->cq;
	qp_init_attr.recv_cq = res->cq;
	qp_init_attr.cap.max_send_wr = 1;
	qp_init_attr.cap.max_recv_wr = 1;
	qp_init_attr.cap.max_send_sge = 1;
	qp_init_attr.cap.max_recv_sge = 1;
	
	int inlineLimit = 512;
	while (inlineLimit >= 1) {
		qp_init_attr.cap.max_inline_data = inlineLimit;
		res->qp = ibv_create_qp(res->pd, &qp_init_attr);
		if (!res->qp) {
			logd("qp set max_inline_data = %d failed, retry", inlineLimit);
			inlineLimit /= 2;
		} else {
			logi("QP set max_inline_data = %d", inlineLimit);
			break;
		}
	}
	res->max_inline_data = inlineLimit;
	
	if (!res->qp)
	{
		qp_init_attr.cap.max_inline_data = 0;
		res->max_inline_data = 0;
		res->qp = ibv_create_qp(res->pd, &qp_init_attr);
		if (!res->qp) {
			loge("failed to create QP");
			rc = 1;
			goto resources_create_exit;
		}
	}
	logi("QP was created, QP number=0x%x", res->qp->qp_num);

	/* 使用标准 mlx5dv API 获取 QP 和 CQ 信息 */
	{
		struct mlx5dv_obj obj = {0};
		struct mlx5dv_qp mlx5dv_qp = {0};
		struct mlx5dv_cq mlx5dv_cq = {0};
		
		/* 设置输入对象 */
		obj.qp.in = res->qp;
		obj.cq.in = res->cq;
		
		/* 设置输出对象指针（需要先分配结构体） */
		obj.qp.out = &mlx5dv_qp;
		obj.cq.out = &mlx5dv_cq;
		
		/* 使用标准 API 初始化对象 */
		if (mlx5dv_init_obj(&obj, MLX5DV_OBJ_QP | MLX5DV_OBJ_CQ)) {
			loge("failed to initialize mlx5dv objects");
			rc = 1;
			goto resources_create_exit;
		}
		
		/* 保存 QP 和 CQ 信息 */
		res->mlx5dv_qp = mlx5dv_qp;
		res->mlx5dv_cq = mlx5dv_cq;
		
		/* 初始化状态变量 */
		res->sq_cur_post = 0;  /* 初始化SQ producer index */
		res->cq_cons_index = 0;  /* 初始化CQ consumer index */
		res->bf_offset = 0;  /* 初始化 Blueflame offset（标准 API 不提供，需要自行维护） */
		
		logi("QP info (standard API): bf.reg=%p, bf.size=%u, sq.buf=%p, sq.wqe_cnt=%u, sq.stride=%u, qp_num=0x%x",
		     res->mlx5dv_qp.bf.reg, res->mlx5dv_qp.bf.size, res->mlx5dv_qp.sq.buf,
		     res->mlx5dv_qp.sq.wqe_cnt, res->mlx5dv_qp.sq.stride, res->qp->qp_num);
		logi("CQ info (standard API): cq_buf=%p, dbrec=%p, cqe_cnt=%u, cqe_size=%u, cqn=0x%x",
		     res->mlx5dv_cq.buf, res->mlx5dv_cq.dbrec, res->mlx5dv_cq.cqe_cnt,
		     res->mlx5dv_cq.cqe_size, res->mlx5dv_cq.cqn);
		
		/* 检查是否支持 Blueflame */
		if (!res->mlx5dv_qp.bf.reg || res->mlx5dv_qp.bf.size == 0) {
			logw("device does not support blueflame (bf.reg=%p, bf.size=%u)", 
			     res->mlx5dv_qp.bf.reg, res->mlx5dv_qp.bf.size);
			rc = 1;
			goto resources_create_exit;
		}
	}

resources_create_exit:
	if (rc)
	{
		if (res->qp)
		{
			ibv_destroy_qp(res->qp);
			res->qp = NULL;
		}
		if (res->mr)
		{
			ibv_dereg_mr(res->mr);
			res->mr = NULL;
		}
		if (res->buf)
		{
			free(res->buf);
			res->buf = NULL;
		}
		if (res->cq)
		{
			ibv_destroy_cq(res->cq);
			res->cq = NULL;
		}
		if (res->pd)
		{
			ibv_dealloc_pd(res->pd);
			res->pd = NULL;
		}
		if (res->ib_ctx)
		{
			ibv_close_device(res->ib_ctx);
			res->ib_ctx = NULL;
		}
		if (dev_list)
		{
			ibv_free_device_list(dev_list);
		}
	}
	return rc;
}

/* 修改QP状态到INIT */
static int modify_qp_to_init(struct ibv_qp *qp)
{
	struct ibv_qp_attr attr;
	int flags;
	int rc;
	memset(&attr, 0, sizeof(attr));
	attr.qp_state = IBV_QPS_INIT;
	attr.port_num = config.ib_port;
	attr.pkey_index = 0;
	attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
	flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
	rc = ibv_modify_qp(qp, &attr, flags);
	if (rc)
		loge("failed to modify QP state to INIT");
	return rc;
}

/* 修改QP状态到RTR（自连接：使用自己的QP信息） */
static int modify_qp_to_rtr(struct ibv_qp *qp, uint32_t qp_num, uint16_t lid, uint8_t *gid)
{
	struct ibv_qp_attr attr;
	int flags;
	int rc;
	memset(&attr, 0, sizeof(attr));
	attr.qp_state = IBV_QPS_RTR;
	attr.path_mtu = IBV_MTU_256;
	attr.dest_qp_num = qp_num;  /* 自连接：使用自己的QP号 */
	attr.rq_psn = 0;
	attr.max_dest_rd_atomic = 1;
	attr.min_rnr_timer = 0x12;
	attr.ah_attr.is_global = 0;
	attr.ah_attr.dlid = lid;  /* 自连接：使用自己的LID */
	attr.ah_attr.sl = 0;
	attr.ah_attr.src_path_bits = 0;
	attr.ah_attr.port_num = config.ib_port;
	if (config.gid_idx >= 0)
	{
		attr.ah_attr.is_global = 1;
		attr.ah_attr.port_num = 1;
		memcpy(&attr.ah_attr.grh.dgid, gid, 16);
		attr.ah_attr.grh.flow_label = 0;
		attr.ah_attr.grh.hop_limit = 1;
		attr.ah_attr.grh.sgid_index = config.gid_idx;
		attr.ah_attr.grh.traffic_class = 0;
	}
	flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
			IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
	rc = ibv_modify_qp(qp, &attr, flags);
	if (rc)
		loge("failed to modify QP state to RTR");
	return rc;
}

/* 修改QP状态到RTS */
static int modify_qp_to_rts(struct ibv_qp *qp)
{
	struct ibv_qp_attr attr;
	int flags;
	int rc;
	memset(&attr, 0, sizeof(attr));
	attr.qp_state = IBV_QPS_RTS;
	attr.timeout = 0x12;
	attr.retry_cnt = 6;
	attr.rnr_retry = 0;
	attr.sq_psn = 0;
	attr.max_rd_atomic = 1;
	flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
			IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
	rc = ibv_modify_qp(qp, &attr, flags);
	if (rc)
		loge("failed to modify QP state to RTS");
	return rc;
}

/* 自连接QP（不需要TCP连接） */
static int connect_qp_self(struct resources *res)
{
	union ibv_gid my_gid;
	int rc = 0;
	
	if (config.gid_idx >= 0)
	{
		rc = ibv_query_gid(res->ib_ctx, config.ib_port, config.gid_idx, &my_gid);
		if (rc)
		{
			loge("could not get gid for port %d, index %d", config.ib_port, config.gid_idx);
			return rc;
		}
	}
	else
		memset(&my_gid, 0, sizeof(my_gid));
	
	logi("Local LID = 0x%x", res->port_attr.lid);
	logi("Local QP number = 0x%x", res->qp->qp_num);
	
	/* 修改QP到INIT状态 */
	rc = modify_qp_to_init(res->qp);
	if (rc)
	{
		loge("change QP state to INIT failed");
		return rc;
	}
	
	/* 修改QP到RTR状态（自连接：使用自己的QP信息） */
	rc = modify_qp_to_rtr(res->qp, res->qp->qp_num, res->port_attr.lid, (uint8_t *)&my_gid);
	if (rc)
	{
		loge("failed to modify QP state to RTR");
		return rc;
	}
	
	/* 修改QP到RTS状态 */
	rc = modify_qp_to_rts(res->qp);
	if (rc)
	{
		loge("failed to modify QP state to RTS");
		return rc;
	}
	
	logi("QP state was changed to RTS (self-connected)");
	return 0;
}

/* RDMA Write操作（自连接）- 手动构建segment并使用doorbell触发 */
static int local_write(struct resources *res)
{
	/* 手动构建RDMA Write的三个segment：ctrl + raddr + data */
	struct mlx5_wqe_ctrl_seg *ctrl;
	struct mlx5_wqe_raddr_seg *raddr;
	struct mlx5_wqe_data_seg *data;
	void *wqe;
	uint64_t remote_addr;
	uint32_t rkey;
	uint32_t data_length;
	uint32_t lkey;
	uint64_t data_addr;
	uint8_t ds;  /* WQE size in 16-byte units */
	uint8_t opcode = MLX5_OPCODE_RDMA_WRITE;
	uint8_t fm_ce_se;
	uint16_t pi = 0;  /* Producer index */
	
	/* 初始化缓冲区 */
	memset(res->buf, 0, config.msg_size);
	memset(res->buf, 8, 4);
	
	logw("before local write:");
	for (int i = 0; i < 8; i++) {
		printf("%d ", res->buf[i]);
	}
	printf("\n");
	
	/* RDMA Write参数 */
	remote_addr = (uint64_t)(uintptr_t)(res->buf + 4);
	rkey = res->mr->rkey;
	data_addr = (uint64_t)(uintptr_t)res->buf;
	data_length = 4;
	lkey = res->mr->lkey;
	
	/* 检查是否支持blueflame */
	if (!res->mlx5dv_qp.bf.reg || res->mlx5dv_qp.bf.size == 0) {
		loge("device does not support blueflame (doorbell), cannot proceed");
		return -1;
	}
	
	/* 
	 * 手动构建WQE segment
	 * 使用标准 API 获取的 QP 信息来计算 WQE 地址和 doorbell 地址
	 */
	{
		/* 计算当前WQE地址：sq.buf + (cur_post % wqe_cnt) * stride */
		uint32_t wqe_idx = res->sq_cur_post % res->mlx5dv_qp.sq.wqe_cnt;
		wqe = (char *)res->mlx5dv_qp.sq.buf + (wqe_idx * res->mlx5dv_qp.sq.stride);
		pi = res->sq_cur_post & 0xffff;  /* 只取低16位 */
		
		logi("Calculated WQE address: wqe=%p, idx=%u, pi=%u", wqe, wqe_idx, pi);
	}
	
	/* 手动构建三个segment */
	ctrl = (struct mlx5_wqe_ctrl_seg *)wqe;
	raddr = (struct mlx5_wqe_raddr_seg *)((char *)wqe + sizeof(struct mlx5_wqe_ctrl_seg));
	data = (struct mlx5_wqe_data_seg *)((char *)raddr + sizeof(struct mlx5_wqe_raddr_seg));
	
	/* 1. 构建ctrl segment */
	ds = (sizeof(struct mlx5_wqe_ctrl_seg) + 
	      sizeof(struct mlx5_wqe_raddr_seg) + 
	      sizeof(struct mlx5_wqe_data_seg)) / 16;
	fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;  /* 需要CQ更新 */
	
	mlx5dv_set_ctrl_seg(ctrl, pi, opcode, 0, res->qp->qp_num, 
	                    fm_ce_se, ds, 0, 0);
	
	/* 2. 构建raddr segment (remote address segment) */
	raddr->raddr = htobe64(remote_addr);
	raddr->rkey = htobe32(rkey);
	raddr->reserved = 0;
	
	/* 3. 构建data segment */
	mlx5dv_set_data_seg(data, data_length, lkey, data_addr);
	
	logi("Manually constructed WQE segments: ctrl=%p, raddr=%p, data=%p", 
	     ctrl, raddr, data);
	logi("ctrl->opmod_idx_opcode=0x%x, ctrl->qpn_ds=0x%x", 
	     be32toh(ctrl->opmod_idx_opcode), be32toh(ctrl->qpn_ds));
	logi("raddr->raddr=0x%" PRIx64 ", raddr->rkey=0x%x", 
	     be64toh(raddr->raddr), be32toh(raddr->rkey));
	logi("data->byte_count=%u, data->lkey=0x%x, data->addr=0x%" PRIx64,
	     be32toh(data->byte_count), be32toh(data->lkey), be64toh(data->addr));
	
	/* 确保内存写入完成（使用内存屏障） */
	__sync_synchronize();
	
	/* 计算doorbell地址：bf.reg + bf_offset（使用自行维护的 bf_offset） */
	void *bf_addr = (char *)res->mlx5dv_qp.bf.reg + res->bf_offset;
	
	/* 通过doorbell触发通信 */
	*((volatile uint64_t *)bf_addr) = *(uint64_t *)ctrl;
	logi("doorbell triggered: bf_addr=%p (bf.reg=%p, bf_offset=0x%x), ctrl=%p", 
	     bf_addr, res->mlx5dv_qp.bf.reg, res->bf_offset, ctrl);
	
	/* 更新SQ producer index（用于下次计算WQE地址） */
	res->sq_cur_post++;
	/* 更新bf_offset（通过异或实现索引环绕，标准 API 不提供，需要自行维护） */
	res->bf_offset ^= res->mlx5dv_qp.bf.size;
	
	if (poll_completion(res)) {
		loge("local write: poll completion failed");
		return -1;
	}
	
	logw("after local write:");
	for (int i = 0; i < 8; i++) {
		printf("%d ", res->buf[i]);
	}
	printf("\n");
	
	return 0;
}

/* RDMA Send/Recv操作（自连接） */
static int local_receive(struct resources *res)
{
	/* 第一步：Post Receive Request */
	{
		struct ibv_recv_wr rr;
		struct ibv_sge sge;
		struct ibv_recv_wr *bad_wr;
		int rc;
		
		/* 准备scatter/gather entry */
		memset(&sge, 0, sizeof(sge));
		sge.addr = (uintptr_t)res->buf + 4;
		sge.length = 4;
		sge.lkey = res->mr->lkey;
		
		/* 初始化缓冲区 */
		memset(res->buf, 0, config.msg_size);
		memset(res->buf, 5, 4);
		
		logw("before local receive:");
		for (int i = 0; i < 8; i++) {
			printf("%d ", res->buf[i]);
		}
		printf("\n");
		
		/* 准备receive work request */
		memset(&rr, 0, sizeof(rr));
		rr.next = NULL;
		rr.wr_id = 0;
		rr.sg_list = &sge;
		rr.num_sge = 1;
		
		/* post Receive Request */
		rc = ibv_post_recv(res->qp, &rr, &bad_wr);
		if (rc)
			loge("failed to post RR");
		else
			logi("Receive Request was posted");
	}
	
	/* 第二步：Post Send Request */
	{
		struct ibv_send_wr sr;
		struct ibv_sge sge;
		struct ibv_send_wr *bad_wr = NULL;
		int rc;
		
		/* 准备scatter/gather entry */
		memset(&sge, 0, sizeof(sge));
		sge.addr = (uintptr_t)res->buf;
		sge.length = 4;
		sge.lkey = res->mr->lkey;
		
		logw("before local send:");
		for (int i = 0; i < 8; i++) {
			printf("%d ", res->buf[i]);
		}
		printf("\n");
		
		/* 准备send work request */
		memset(&sr, 0, sizeof(sr));
		sr.next = NULL;
		sr.wr_id = 0;
		sr.sg_list = &sge;
		sr.num_sge = 1;
		sr.opcode = IBV_WR_SEND;
		sr.send_flags = IBV_SEND_SIGNALED;
		
		rc = ibv_post_send(res->qp, &sr, &bad_wr);
		if (rc)
			loge("failed to post SR");
		
		/* recv任务先提交，先 poll recv任务 */
		if (poll_completion(res)) {
			loge("local recv: poll completion failed");
		}
		
		/* send 任务后提交，后 poll send 任务 */
		if (poll_completion(res)) {
			loge("local send: poll completion failed");
		}
		
		logw("after local send/receive:");
		for (int i = 0; i < 8; i++) {
			printf("%d ", res->buf[i]);
		}
		printf("\n");
	}
	
	return 0;
}

/* 销毁资源 */
static int resources_destroy(struct resources *res)
{
	int rc = 0;
	if (res->qp)
		if (ibv_destroy_qp(res->qp))
		{
			loge("failed to destroy QP");
			rc = 1;
		}
	if (res->mr)
		if (ibv_dereg_mr(res->mr))
		{
			loge("failed to deregister MR");
			rc = 1;
		}
	if (res->buf)
		free(res->buf);
	if (res->cq)
		if (ibv_destroy_cq(res->cq))
		{
			loge("failed to destroy CQ");
			rc = 1;
		}
	if (res->pd)
		if (ibv_dealloc_pd(res->pd))
		{
			loge("failed to deallocate PD");
			rc = 1;
		}
	if (res->ib_ctx)
		if (ibv_close_device(res->ib_ctx))
		{
			loge("failed to close device context");
			rc = 1;
		}
	return rc;
}

/* 打印配置信息 */
static void print_config(void)
{
	fprintf(stdout, " ------------------------------------------------\n");
	fprintf(stdout, " Device name : \"%s\"\n", config.dev_name ? config.dev_name : "auto");
	fprintf(stdout, " IB port : %u\n", config.ib_port);
	if (config.gid_idx >= 0)
		fprintf(stdout, " GID index : %u\n", config.gid_idx);
	fprintf(stdout, " Message size : %zu bytes\n", config.msg_size);
	fprintf(stdout, " Repeat count : %d\n", config.repeat_count);
	fprintf(stdout, " ------------------------------------------------\n\n");
}

int main(int argc, char *argv[])
{
	struct resources res;
	int rc = 1;
	
	/* 解析命令行参数（简化版） */
	if (argc > 1) {
		for (int i = 1; i < argc; i++) {
			if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
				config.dev_name = strdup(argv[++i]);
			} else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
				config.ib_port = atoi(argv[++i]);
			} else if (strcmp(argv[i], "-g") == 0 && i + 1 < argc) {
				config.gid_idx = atoi(argv[++i]);
			} else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
				config.msg_size = atoi(argv[++i]);
			} else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
				config.repeat_count = atoi(argv[++i]);
				if (config.repeat_count <= 0) {
					fprintf(stderr, "Error: repeat count must be > 0\n");
					return 1;
				}
			} else if (strcmp(argv[i], "-h") == 0) {
				printf("Usage: %s [-d device] [-i ib_port] [-g gid_idx] [-s msg_size] [-r repeat_count]\n", argv[0]);
				return 0;
			}
		}
	}
	
	print_config();
	
	/* 初始化资源 */
	resources_init(&res);
	
	/* 创建资源 */
	if (resources_create(&res))
	{
		loge("failed to create resources");
		goto main_exit;
	}
	
	/* 自连接QP（不需要TCP连接） */
	if (connect_qp_self(&res))
	{
		loge("failed to connect QP (self-connection)");
		goto main_exit;
	}
	
	logi("Starting RDMA self-connection test...");
	
	/* 测试RDMA Write（重复r次） */
	logi("=== Testing RDMA Write (repeat %d times) ===", config.repeat_count);
	for (int i = 0; i < config.repeat_count; i++) {
		if (config.repeat_count > 1) {
			logw("--- RDMA Write test iteration %d/%d ---", i + 1, config.repeat_count);
		}
		if (local_write(&res)) {
			loge("RDMA Write test failed at iteration %d/%d", i + 1, config.repeat_count);
			goto main_exit;
		}
	}
	
	logi("All tests completed successfully!");
	rc = 0;

main_exit:
	if (resources_destroy(&res))
	{
		loge("failed to destroy resources");
		rc = 1;
	}
	if (config.dev_name)
		free((char *)config.dev_name);
	
	logi("test result is %d", rc);
	return rc;
}

/*
在一台机器上即可运行，不需要两台机器。

~/jinbo/ib$ ./manual_trigger -d ibp132s0 -r 1 -s 1600
 ------------------------------------------------
 Device name : "ibp132s0"
 IB port : 1
 Message size : 1600 bytes
 ------------------------------------------------

[2025-12-24 09:26:19][INFO ][manual_trigger.c:resources_create:132] searching for IB devices in host
[2025-12-24 09:26:19][INFO ][manual_trigger.c:resources_create:148] found 1 device(s)
[2025-12-24 09:26:19][INFO ][manual_trigger.c:resources_create:232] MR was registered with addr=0x21f490b0, lkey=0x1ff0af, rkey=0x1ff0af
[2025-12-24 09:26:19][INFO ][manual_trigger.c:resources_create:253] QP set max_inline_data = 512
[2025-12-24 09:26:19][INFO ][manual_trigger.c:resources_create:270] QP was created, QP number=0x12d
[2025-12-24 09:26:19][INFO ][manual_trigger.c:connect_qp_self:406] Local LID = 0x1
[2025-12-24 09:26:19][INFO ][manual_trigger.c:connect_qp_self:407] Local QP number = 0x12d
[2025-12-24 09:26:19][INFO ][manual_trigger.c:connect_qp_self:433] QP state was changed to RTS (self-connected)
[2025-12-24 09:26:19][INFO ][manual_trigger.c:main:688] Starting RDMA self-connection test...
[2025-12-24 09:26:19][INFO ][manual_trigger.c:main:691] === Testing RDMA Write ===
[2025-12-24 09:26:19][WARN ][manual_trigger.c:local_write:456] before local write:
8 8 8 8 0 0 0 0 
[2025-12-24 09:26:19][INFO ][manual_trigger.c:local_write:482] local write was posted
[2025-12-24 09:26:19][INFO ][manual_trigger.c:poll_completion:103] completion was found in CQ with status 0x0, opcode=1 (RDMA_WRITE)
[2025-12-24 09:26:19][WARN ][manual_trigger.c:local_write:489] after local write:
8 8 8 8 8 8 8 8 
[2025-12-24 09:26:19][INFO ][manual_trigger.c:main:698] === Testing RDMA Send/Recv ===
[2025-12-24 09:26:19][WARN ][manual_trigger.c:local_receive:518] before local receive:
5 5 5 5 0 0 0 0 
[2025-12-24 09:26:19][INFO ][manual_trigger.c:local_receive:536] Receive Request was posted
[2025-12-24 09:26:19][WARN ][manual_trigger.c:local_receive:552] before local send:
5 5 5 5 0 0 0 0 
[2025-12-24 09:26:19][INFO ][manual_trigger.c:poll_completion:103] completion was found in CQ with status 0x0, opcode=128 (RECV)
[2025-12-24 09:26:19][INFO ][manual_trigger.c:poll_completion:103] completion was found in CQ with status 0x0, opcode=0 (SEND)
[2025-12-24 09:26:19][WARN ][manual_trigger.c:local_receive:585] after local send/receive:
5 5 5 5 5 5 5 5 
[2025-12-24 09:26:19][INFO ][manual_trigger.c:main:704] All tests completed successfully!
[2025-12-24 09:26:19][INFO ][manual_trigger.c:main:716] test result is 0

*/

