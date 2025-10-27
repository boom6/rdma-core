/*
* BUILD COMMAND:
$ bash build.sh

RUN COMMAND:
$ cd build/bin
$ sudo env VERBS_LOG_LEVEL=2 ./uar_test 127.0.0.1 -c ../../libibverbs/examples/cudaPlugin.so  -r 1000 -p 12345
$ ./uar_test -p 12345 -r 1000

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <inttypes.h>
#include <endian.h>
#include <byteswap.h>
#include <getopt.h>
#include <pthread.h>
#include <dlfcn.h>
#include "log.h"

#include <sys/time.h>
#include <arpa/inet.h>
#include "../verbs.h"
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

// 生产者线程和消费者线程都用 blueflame 触发通信的方式，
#define MULTI_THREAD_TEST_USE_BF 1
// 在main函数里 post send，然后立即 poll_completion 的方式，
// 通过 NORMAL_TEST_USE_BF 控制是用 blueflame 触发通信还是在 verbs 接口内部触发
// NORMAL_TEST_USE_BF 仅在 MULTI_THREAD_TEST_USE_BF 为 0 时有效
#define NORMAL_TEST_USE_BF 1

// 控制是否打印 buffer 的内容
#define DEBUG_ENABLE_PRINT_DATA 0

#define MAX_POLL_CQ_TIMEOUT 5000
#define MAX_WR_COUNT 256
#define DEFAULT_MSG_SIZE 64 
#define EVEN_BF_OFFSET 0x800
#define ODD_BF_OFFSET 0x900
#define BF_SIZE 0x1000
#define CTRL_SIZE (MAX_WR_COUNT * 4 * 64)
#if __BYTE_ORDER == __LITTLE_ENDIAN
static inline uint64_t htonll(uint64_t x) { return bswap_64(x); }
static inline uint64_t ntohll(uint64_t x) { return bswap_64(x); }
#elif __BYTE_ORDER == __BIG_ENDIAN
static inline uint64_t htonll(uint64_t x) { return x; }
static inline uint64_t ntohll(uint64_t x) { return x; }
#else
#error __BYTE_ORDER is neither __LITTLE_ENDIAN nor __BIG_ENDIAN
#endif

#define MAX_DEVX_INFO_COUNT (MAX_WR_COUNT)
// #define MAX_DEVX_INFO_COUNT (16)
#define MAX_POLL_COUNT MAX_WR_COUNT
// request_cnt 与 complete_cnt 相等，则说明 devx_infos 为空
// 注意：不使用求余操作，直接递增，使用volatile确保多线程可见性
static volatile uint64_t request_cnt = 0;  // 已请求的任务个数
static volatile uint64_t trigger_cnt = 0;  // 已触发的任务个数
static volatile uint64_t complete_cnt = 0; // 已完成的任务个数
static void* bf_base = NULL;
static void* ctrl_base = NULL;
uint32_t ctrl_offsets[MAX_DEVX_INFO_COUNT] = {0};
static volatile int thread_running = 1; // 控制线程运行状态
static volatile int monitor_running = 1; // 控制监控线程运行状态

// GPU模式相关变量
static void *cuda_plugin_handle = NULL;
static int (*init_cuda_func)(void) = NULL;
static int (*cleanup_cuda_func)(void) = NULL;
static int (*convert_host_va_to_gpu_va_func)(void *, size_t, int, void **) = NULL;
static int (*trigger_doorbell_func)(void *, void *, uint32_t) = NULL;
static int (*unregister_host_va_func)(void *) = NULL;

// GPU端指针
static void *gpu_bf = NULL;
static void *gpu_ctrl = NULL;

/* structure of test parameters */
struct config_t
{
	const char *dev_name; /* IB device name */
	char *server_name;	/* server host name */
	u_int32_t tcp_port;   /* server TCP port */
	int ib_port;		  /* local IB port to work with */
	int gid_idx;		  /* gid index to use */
	int repeat;		  /* repeat count */
	int mode;		  /* trigger mode: 0=CPU, 1=GPU */
	char cuda_plugin_path[256]; /* CUDA plugin path */
	size_t msg_size;	  /* message buffer size */
};
/* structure to exchange data which is needed to connect the QPs */
struct cm_con_data_t
{
	uint64_t addr;   /* Buffer address */
	uint32_t rkey;   /* Remote key */
	uint32_t qp_num; /* QP number */
	uint16_t lid;	/* LID of the IB port */
	uint8_t gid[16]; /* gid */
} __attribute__((packed));

struct hca_attr_t {
    uint32_t max_inline_data;
} __attribute__((packed));

/* structure of system resources */
struct resources
{
	struct ibv_device_attr
		device_attr;
	/* Device attributes */
	struct ibv_port_attr port_attr;	/* IB port attributes */
	struct hca_attr_t hca_attr;     /* HCA transport attributes */
	struct cm_con_data_t remote_props; /* values to connect to remote side */
	struct ibv_context *ib_ctx;		   /* device handle */
	struct ibv_pd *pd;				   /* PD handle */
	struct ibv_cq *cq;				   /* CQ handle */
	struct ibv_qp *qp;				   /* QP handle */
	struct ibv_mr *mr;				   /* MR handle for buf */
	char *buf;						   /* memory buffer pointer, used for RDMA and send
ops */
	int sock;						   /* TCP socket file descriptor */
	char sync[2];
	void *bf;
};
struct config_t config = {
	NULL,  /* dev_name */
	NULL,  /* server_name */
	19875, /* tcp_port */
	1,	 /* ib_port */
	-1, /* gid_idx */
	1, /* repeat */
	0,  /* mode: 0=CPU, 1=GPU */
	"./cudaPlugin.so",  /* cuda_plugin_path */
#if MULTI_THREAD_TEST_USE_BF
	DEFAULT_MSG_SIZE  /* msg_size */
#else
	8388608 /* msg_size 为 8MB，用于测试 2B~8MB 的传输 */
#endif
};

static int sock_connect(const char *servername, int port)
{
	struct addrinfo *resolved_addr = NULL;
	struct addrinfo *iterator;
	char service[6];
	int sockfd = -1;
	int listenfd = 0;
	int tmp;
	struct addrinfo hints =
		{
			.ai_flags = AI_PASSIVE,
			.ai_family = AF_INET,
			.ai_socktype = SOCK_STREAM};
	if (sprintf(service, "%d", port) < 0)
		goto sock_connect_exit;
	/* Resolve DNS address, use sockfd as temp storage */
	sockfd = getaddrinfo(servername, service, &hints, &resolved_addr);
	if (sockfd < 0)
	{
		loge("%s for %s:%d", gai_strerror(sockfd), servername, port);
		goto sock_connect_exit;
	}
	/* Search through results and find the one we want */
	for (iterator = resolved_addr; iterator; iterator = iterator->ai_next)
	{
		sockfd = socket(iterator->ai_family, iterator->ai_socktype, iterator->ai_protocol);
		if (sockfd >= 0)
		{
			/* Set socket reuse option to avoid TIME_WAIT issues */
			int optval = 1;
			if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)) < 0) {
				loge("failed to set SO_REUSEADDR");
				close(sockfd);
				sockfd = -1;
				continue;
			}
			
			if (servername){
				/* Client mode. Initiate connection to remote */
				if ((tmp = connect(sockfd, iterator->ai_addr, iterator->ai_addrlen)))
				{
					loge("failed connect");
					close(sockfd);
					sockfd = -1;
				}
            }
			else
			{
					/* Server mode. Set up listening socket an accept a connection */
					listenfd = sockfd;
					sockfd = -1;
					if (bind(listenfd, iterator->ai_addr, iterator->ai_addrlen))
						goto sock_connect_exit;
					listen(listenfd, 1);
					sockfd = accept(listenfd, NULL, 0);
			}
		}
	}
sock_connect_exit:
	if (listenfd)
		close(listenfd);
	if (resolved_addr)
		freeaddrinfo(resolved_addr);
	if (sockfd < 0)
	{
		if (servername)
			loge("Couldn't connect to %s:%d", servername, port);
		else
		{
			perror("server accept");
			loge("accept() failed");
		}
	}
	return sockfd;
}

static int sock_sync_data(int sock, int xfer_size, char *local_data, char *remote_data)
{
	int rc;
	int read_bytes = 0;
	int total_read_bytes = 0;
	rc = write(sock, local_data, xfer_size);
	if (rc < xfer_size)
		loge("Failed writing data during sock_sync_data");
	else
		rc = 0;
	while (!rc && total_read_bytes < xfer_size)
	{
		read_bytes = read(sock, remote_data, xfer_size);
		if (read_bytes > 0)
			total_read_bytes += read_bytes;
		else
			rc = read_bytes;
	}
	return rc;
}

static int poll_completion(struct resources *res)
{
	struct ibv_wc wc;
	unsigned long start_time_msec;
	unsigned long cur_time_msec;
	struct timeval cur_time;
	int poll_result;
	int rc = 0;
	/* poll the completion for a while before giving up of doing it .. */
	gettimeofday(&cur_time, NULL);
	start_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
	do
	{
		poll_result = ibv_poll_cq(res->cq, 1, &wc);
		gettimeofday(&cur_time, NULL);
		cur_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
	} while ((poll_result == 0) && ((cur_time_msec - start_time_msec) < MAX_POLL_CQ_TIMEOUT));
	if (poll_result < 0)
	{
		/* poll CQ failed */
		loge("poll CQ failed");
		rc = 1;
	}
	else if (poll_result == 0)
	{ /* the CQ is empty */
		loge("completion wasn't found in the CQ after timeout");
		rc = 1;
	}
	else
	{
		/* CQE found */
		logi("completion was found in CQ with status 0x%x", wc.status);
		/* check the completion status (here we don't care about the completion opcode */
		if (wc.status != IBV_WC_SUCCESS)
		{
			loge("got bad completion with status: 0x%x, vendor syndrome: 0x%x", wc.status,
					wc.vendor_err);
			rc = 1;
		}
	}
	return rc;
}

/**
 * 轮询完成队列，返回是否出错（0表示成功，1表示失败）
 * completed_count: 输出参数，返回实际完成的任务个数
 * need_count: 期望获取到的完成个数，实际完成的数量可能小于期望的
 */
static int poll_completion_once(struct resources *res, uint32_t *completed_count, uint32_t need_count)
{
	struct ibv_wc wc_array[need_count]; // 动态分配数组来存储多个完成事件
	int poll_result;
	int rc = 0;
	
	*completed_count = 0;
	
	// 一次性轮询need_count个完成事件
	poll_result = ibv_poll_cq(res->cq, need_count, wc_array);
	
	if (poll_result < 0) {
		/* poll CQ failed */
		loge("poll CQ failed in poll_completion_once");
		rc = 1;
	} else if (poll_result == 0) {
		/* the CQ is empty */
		*completed_count = 0;
	} else {
		/* CQEs found */
		*completed_count = poll_result;
		logi("poll_completion_once found %d completions", poll_result);
		
		// 检查所有完成事件的状态
		for (int i = 0; i < poll_result; i++) {
			if (wc_array[i].status != IBV_WC_SUCCESS) {
				loge("got bad completion with status: 0x%x, vendor syndrome: 0x%x", 
					 wc_array[i].status, wc_array[i].vendor_err);
				rc = 1;
				break;
			}
		}
	}
	
	return rc;
}

/**
 * 加载CUDA插件
 */
static int load_cuda_plugin(void) {
    if (config.mode == 0) {
        	logi("CPU mode, skipping CUDA plugin loading");
        return 0;
    }
    
    	logi("Loading CUDA plugin for GPU mode...");
    
    // 加载CUDA插件
    cuda_plugin_handle = dlopen(config.cuda_plugin_path, RTLD_LAZY);
    if (!cuda_plugin_handle) {
        		loge("Failed to load %s: %s", config.cuda_plugin_path, dlerror());
        return -1;
    }
    
    // 获取函数指针
    init_cuda_func = (int (*)(void))dlsym(cuda_plugin_handle, "init_cuda");
    if (!init_cuda_func) {
        		loge("Failed to get init_cuda function: %s", dlerror());
        dlclose(cuda_plugin_handle);
        return -1;
    }
    
    cleanup_cuda_func = (int (*)(void))dlsym(cuda_plugin_handle, "cleanup_cuda");
    if (!cleanup_cuda_func) {
        		loge("Failed to get cleanup_cuda function: %s", dlerror());
        dlclose(cuda_plugin_handle);
        return -1;
    }
    
    convert_host_va_to_gpu_va_func = (int (*)(void *, size_t, int, void **))dlsym(cuda_plugin_handle, "ConvertHostVA2GpuVA");
    if (!convert_host_va_to_gpu_va_func) {
        		loge("Failed to get ConvertHostVA2GpuVA function: %s", dlerror());
        dlclose(cuda_plugin_handle);
        return -1;
    }
    
    trigger_doorbell_func = (int (*)(void *, void *, uint32_t))dlsym(cuda_plugin_handle, "TriggerDoorbell");
    if (!trigger_doorbell_func) {
        		loge("Failed to get TriggerDoorbell function: %s", dlerror());
        dlclose(cuda_plugin_handle);
        return -1;
    }
    
    unregister_host_va_func = (int (*)(void *))dlsym(cuda_plugin_handle, "UnregisterHostVA");
    if (!unregister_host_va_func) {
        		loge("Failed to get UnregisterHostVA function: %s", dlerror());
        dlclose(cuda_plugin_handle);
        return -1;
    }
    
    // 初始化CUDA环境
    if (init_cuda_func() != 0) {
        		loge("Failed to initialize CUDA environment");
        dlclose(cuda_plugin_handle);
        return -1;
    }
    
    	logi("CUDA plugin loaded successfully");
    return 0;
}

static int convert_blueflame_to_gpu(void *host_bf, void *host_ctrl) {
    if (!convert_host_va_to_gpu_va_func) {
        loge("ConvertHostVA2GpuVA function not available");
        return -1;
    }
    
    logi("Converting first devx_info to GPU: bf=%p, ctrl=%p", host_bf, host_ctrl);
    
    // 转换bf指针 (type=0)
    if (convert_host_va_to_gpu_va_func(host_bf, BF_SIZE, 0, &gpu_bf) != 0) {
        loge("Failed to convert bf to GPU VA");
        return -1;
    }
    
    // 转换ctrl指针 (type=1)
    if (convert_host_va_to_gpu_va_func(host_ctrl, CTRL_SIZE, 1, &gpu_ctrl) != 0) {
        loge("Failed to convert ctrl to GPU VA");
        return -1;
    }
    
    logi("Successfully converted to GPU: bf=%p->%p, ctrl=%p->%p",
		   host_bf, gpu_bf, host_ctrl, gpu_ctrl);
    return 0;
}

/**
 * 卸载CUDA插件
 */
static void unload_cuda_plugin(void) {
    if (cuda_plugin_handle) {
        // 取消注册GPU指针
        if (unregister_host_va_func) {
            if (gpu_bf) {
                unregister_host_va_func(gpu_bf);
                gpu_bf = NULL;
            }
            if (gpu_ctrl) {
                unregister_host_va_func(gpu_ctrl);
                gpu_ctrl = NULL;
            }
        }
        
        if (cleanup_cuda_func) {
            cleanup_cuda_func();
        }
        dlclose(cuda_plugin_handle);
        cuda_plugin_handle = NULL;
        	logi("CUDA plugin unloaded");
    }
}

static int post_send(struct resources *res, int opcode, struct ibv_devx_info *devx_info)
{
	struct ibv_send_wr sr;
	struct ibv_sge sge;
	struct ibv_send_wr *bad_wr = NULL;
	int rc;
	/* prepare the scatter/gather entry */
	memset(&sge, 0, sizeof(sge));
	sge.addr = (uintptr_t)res->buf;
	sge.length = config.msg_size;
	sge.lkey = res->mr->lkey;
	/* prepare the send work request */
	memset(&sr, 0, sizeof(sr));
	sr.next = NULL;
	sr.wr_id = 0;
	sr.sg_list = &sge;
	sr.num_sge = 1;
	sr.opcode = opcode;
	if (config.msg_size <= res->hca_attr.max_inline_data) {
		sr.send_flags |= IBV_SEND_INLINE;
	}
	sr.send_flags = IBV_SEND_SIGNALED;
	if (opcode != IBV_WR_SEND)
	{
		sr.wr.rdma.remote_addr = res->remote_props.addr;
		sr.wr.rdma.rkey = res->remote_props.rkey;
	}
	/* there is a Receive Request in the responder side, so we won't get any into RNR flow */
#if NORMAL_TEST_USE_BF
	rc = ibv_devx_post_send(res->qp, &sr, &bad_wr, devx_info);
#else
	rc = ibv_post_send(res->qp, &sr, &bad_wr);
#endif
	if (rc)
		loge("failed to post SR");

	return rc;
}

__attribute__((__unused__)) static int post_receive(struct resources *res)
{
	struct ibv_recv_wr rr;
	struct ibv_sge sge;
	struct ibv_recv_wr *bad_wr;
	int rc;
	/* prepare the scatter/gather entry */
	memset(&sge, 0, sizeof(sge));
	sge.addr = (uintptr_t)res->buf;
	sge.length = config.msg_size;
	sge.lkey = res->mr->lkey;
	/* prepare the receive work request */
	memset(&rr, 0, sizeof(rr));
	rr.next = NULL;
	rr.wr_id = 0;
	rr.sg_list = &sge;
	rr.num_sge = 1;
	/* post the Receive Request to the RQ */
	rc = ibv_post_recv(res->qp, &rr, &bad_wr);
	if (rc)
		loge("failed to post RR");
	else
		logi("Receive Request was posted");
	return rc;
}

static void resources_init(struct resources *res)
{
	memset(res, 0, sizeof *res);
	res->sock = -1;
}

static int resources_create(struct resources *res)
{
	struct ibv_device **dev_list = NULL;
	struct ibv_qp_init_attr qp_init_attr;
	struct ibv_device *ib_dev = NULL;
	size_t size;
	int i;
	int mr_flags = 0;
	int cq_size = 0;
	int num_devices;
	int rc = 0;
	/* if client side */
	if (config.server_name)
	{
		res->sock = sock_connect(config.server_name, config.tcp_port);
		if (res->sock < 0)
		{
			loge("failed to establish TCP connection to server %s, port %d",
					config.server_name, config.tcp_port);
			rc = -1;
			goto resources_create_exit;
		}
	}
	else
	{
		logi("waiting on port %d for TCP connection", config.tcp_port);
		res->sock = sock_connect(NULL, config.tcp_port);
		if (res->sock < 0)
		{
			loge("failed to establish TCP connection with client on port %d",
					config.tcp_port);
			rc = -1;
			goto resources_create_exit;
		}
	}
	logi("TCP connection was established");
	logi("searching for IB devices in host");
	/* get device names in the system */
	dev_list = ibv_get_device_list(&num_devices);
	if (!dev_list)
	{
		loge("failed to get IB devices list");
		rc = 1;
		goto resources_create_exit;
	}
	/* if there isn't any IB device in host */
	if (!num_devices)
	{
		loge("found %d device(s)", num_devices);
		rc = 1;
		goto resources_create_exit;
	}
	logi("found %d device(s)", num_devices);
	/* search for the specific device we want to work with */
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
	/* if the device wasn't found in host */
	if (!ib_dev)
	{
		loge("IB device %s wasn't found", config.dev_name);
		rc = 1;
		goto resources_create_exit;
	}
	/* get device handle */
	res->ib_ctx = ibv_open_device(ib_dev);
	if (!res->ib_ctx)
	{
		loge("failed to open device %s", config.dev_name);
		rc = 1;
		goto resources_create_exit;
	}
	/* We are now done with device list, free it */
	ibv_free_device_list(dev_list);
	dev_list = NULL;
	ib_dev = NULL;
	/* query port properties */
	if (ibv_query_port(res->ib_ctx, config.ib_port, &res->port_attr))
	{
		loge("ibv_query_port on port %u failed", config.ib_port);
		rc = 1;
		goto resources_create_exit;
	}
	/* allocate Protection Domain */
	res->pd = ibv_alloc_pd(res->ib_ctx);
	if (!res->pd)
	{
		loge("ibv_alloc_pd failed");
		rc = 1;
		goto resources_create_exit;
	}
	/* each side will send only one WR, so Completion Queue with 1 entry is enough */
	cq_size = MAX_WR_COUNT;
	res->cq = ibv_create_cq(res->ib_ctx, cq_size, NULL, NULL, 0);
	if (!res->cq)
	{
		loge("failed to create CQ with %u entries", cq_size);
		rc = 1;
		goto resources_create_exit;
	}
	/* allocate the memory buffer that will hold the data */
	size = config.msg_size;
	res->buf = (char *)malloc(size);
	if (!res->buf)
	{
		loge("failed to malloc %Zu bytes to memory buffer", size);
		rc = 1;
		goto resources_create_exit;
	}
	memset(res->buf, 0, size);

	/* register the memory buffer */
	mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
	res->mr = ibv_reg_mr(res->pd, res->buf, size, mr_flags);
	if (!res->mr)
	{
		loge("ibv_reg_mr failed with mr_flags=0x%x", mr_flags);
		rc = 1;
		goto resources_create_exit;
	}
	logi("MR was registered with addr=%p, lkey=0x%x, rkey=0x%x, flags=0x%x",
			res->buf, res->mr->lkey, res->mr->rkey, mr_flags);
	/* create the Queue Pair */
	memset(&qp_init_attr, 0, sizeof(qp_init_attr));
	qp_init_attr.qp_type = IBV_QPT_RC;
	qp_init_attr.sq_sig_all = 1;
	qp_init_attr.send_cq = res->cq;
	qp_init_attr.recv_cq = res->cq;
	qp_init_attr.cap.max_send_wr = MAX_WR_COUNT;
	qp_init_attr.cap.max_recv_wr = MAX_WR_COUNT;
	qp_init_attr.cap.max_send_sge = 1;
	qp_init_attr.cap.max_recv_sge = 1;

	int inlineLimit = 512;
	while (inlineLimit >= 1) {
        qp_init_attr.cap.max_inline_data = inlineLimit;
        res->qp = ibv_create_qp(res->pd, &qp_init_attr);
        if (!res->qp) {
            logd("qp set max_inline_data = %lu failed, retry, errno = %s", inlineLimit, strerror(errno));
            inlineLimit /= 2;
        } else {
            logi("QP set max_inline_data = %lu", inlineLimit);
            break;
        }
    }
    res->hca_attr.max_inline_data = inlineLimit;

	if (!res->qp)
	{
		qp_init_attr.cap.max_inline_data = 0;
		res->hca_attr.max_inline_data = 0;
		res->qp = ibv_create_qp(res->pd, &qp_init_attr);
		if (!res->qp) {
			loge("failed to create QP");
			rc = 1;
			goto resources_create_exit;
		}
	}
	logi("QP was created, QP number=0x%x", res->qp->qp_num);

	if (ibv_has_custom_features(res->ib_ctx)) {
		res->bf = ibv_get_blueflame(res->qp);
		if (!res->bf) {
			loge("failed to get BF");
			rc = 1;
			goto resources_create_exit;
		}
		logi("BF was got, BF address=%p", res->bf);
	}

resources_create_exit:
	if (rc)
	{
		/* Error encountered, cleanup */
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
			dev_list = NULL;
		}
		if (res->sock >= 0)
		{
					if (close(res->sock))
			loge("failed to close socket");
			res->sock = -1;
		}
	}
	return rc;
}

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

static int modify_qp_to_rtr(struct ibv_qp *qp, uint32_t remote_qpn, uint16_t dlid, uint8_t *dgid)
{
	struct ibv_qp_attr attr;
	int flags;
	int rc;
	memset(&attr, 0, sizeof(attr));
	attr.qp_state = IBV_QPS_RTR;
	attr.path_mtu = IBV_MTU_256;
	attr.dest_qp_num = remote_qpn;
	attr.rq_psn = 0;
	attr.max_dest_rd_atomic = 1;
	attr.min_rnr_timer = 0x12;
	attr.ah_attr.is_global = 0;
	attr.ah_attr.dlid = dlid;
	attr.ah_attr.sl = 0;
	attr.ah_attr.src_path_bits = 0;
	attr.ah_attr.port_num = config.ib_port;
	if (config.gid_idx >= 0)
	{
		attr.ah_attr.is_global = 1;
		attr.ah_attr.port_num = 1;
		memcpy(&attr.ah_attr.grh.dgid, dgid, 16);
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

static int connect_qp(struct resources *res)
{
	struct cm_con_data_t local_con_data;
	struct cm_con_data_t remote_con_data;
	struct cm_con_data_t tmp_con_data;
	int rc = 0;
	char temp_char;
	union ibv_gid my_gid;
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
		memset(&my_gid, 0, sizeof my_gid);
	/* exchange using TCP sockets info required to connect QPs */
	local_con_data.addr = htonll((uintptr_t)res->buf);
	local_con_data.rkey = htonl(res->mr->rkey);
	local_con_data.qp_num = htonl(res->qp->qp_num);
	local_con_data.lid = htons(res->port_attr.lid);
	memcpy(local_con_data.gid, &my_gid, 16);
	logi("Local LID = 0x%x", res->port_attr.lid);
	if (sock_sync_data(res->sock, sizeof(struct cm_con_data_t), (char *)&local_con_data, (char *)&tmp_con_data) < 0)
	{
		loge("failed to exchange connection data between sides");
		rc = 1;
		goto connect_qp_exit;
	}
	remote_con_data.addr = ntohll(tmp_con_data.addr);
	remote_con_data.rkey = ntohl(tmp_con_data.rkey);
	remote_con_data.qp_num = ntohl(tmp_con_data.qp_num);
	remote_con_data.lid = ntohs(tmp_con_data.lid);
	memcpy(remote_con_data.gid, tmp_con_data.gid, 16);
	/* save the remote side attributes, we will need it for the post SR */
	res->remote_props = remote_con_data;
	logi("Remote address = 0x%" PRIx64, remote_con_data.addr);
	logi("Remote rkey = 0x%x", remote_con_data.rkey);
	logi("Remote QP number = 0x%x", remote_con_data.qp_num);
	logi("Remote LID = 0x%x", remote_con_data.lid);
	if (config.gid_idx >= 0)
	{
		uint8_t *p = remote_con_data.gid;
		logi("Remote GID =%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x",p[0],
				  p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15]);
	}
	/* modify the QP to init */
	rc = modify_qp_to_init(res->qp);
	if (rc)
	{
		loge("change QP state to INIT failed");
		goto connect_qp_exit;
	}

	/* modify the QP to RTR */
	rc = modify_qp_to_rtr(res->qp, remote_con_data.qp_num, remote_con_data.lid, remote_con_data.gid);
	if (rc)
	{
		loge("failed to modify QP state to RTR");
		goto connect_qp_exit;
	}
	rc = modify_qp_to_rts(res->qp);
	if (rc)
	{
		loge("failed to modify QP state to RTR");
		goto connect_qp_exit;
	}
	logi("QP state was change to RTS");
	/* sync to make sure that both sides are in states that they can connect to prevent packet loose */
	if (sock_sync_data(res->sock, 1, res->sync, &temp_char)) /* just send a dummy char back and forth */
	{
		loge("sync error after QPs are were moved to RTS");
		rc = 1;
	}
connect_qp_exit:
	return rc;
}

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
	if (res->sock >= 0)
		if (close(res->sock))
		{
			loge("failed to close socket");
			rc = 1;
		}
	return rc;
}

static void print_config(void)
{
	fprintf(stdout, " ------------------------------------------------\n");
	fprintf(stdout, " Device name : \"%s\"\n", config.dev_name);
	fprintf(stdout, " IB port : %u\n", config.ib_port);
	if (config.server_name)
		fprintf(stdout, " IP : %s\n", config.server_name);
	fprintf(stdout, " TCP port : %u\n", config.tcp_port);
	if (config.gid_idx >= 0)
		fprintf(stdout, " GID index : %u\n", config.gid_idx);
	fprintf(stdout, " Message size : %zu bytes\n", config.msg_size);
	fprintf(stdout, " ------------------------------------------------\n\n");
}


static void usage(const char *argv0)
{
	fprintf(stdout, "Usage:\n");
	fprintf(stdout, " %s start a server and wait for connection\n", argv0);
	fprintf(stdout, " %s <host> connect to server at <host>\n", argv0);
	fprintf(stdout, "\n");
	fprintf(stdout, "Options:\n");
	fprintf(stdout, " -p, --port <port> listen on/connect to port <port> (default 18515)\n");
	fprintf(stdout, " -d, --ib-dev <dev> use IB device <dev> (default first device found)\n");
	fprintf(stdout, " -i, --ib-port <port> use port <port> of IB device (default 1)\n");
	fprintf(stdout, " -g, --gid_idx <git index> gid index to be used in GRH (default not used)\n");
	fprintf(stdout, " -r, --repeat <repeat> repeat count (default 1)\n");
	fprintf(stdout, " -m, --mode <mode> trigger mode: 0=CPU, 1=GPU (default 0)\n");
	fprintf(stdout, " -c, --cuda-plugin <path> CUDA plugin path (default ./cudaPlugin.so)\n");
	fprintf(stdout, " -s, --size <size> message buffer size in bytes (default %d)\n", DEFAULT_MSG_SIZE);
}

static void* producer_thread(void *arg)
{
	struct resources *res = (struct resources *)arg;

	// 获取 bf 与 ctrl 基地址，以及通信预热
	{
		// bf 有两个，even bf 偏移为 0x800，odd bf 偏移为 0x900，交替使用
		struct ibv_devx_info devx_info;
		if (post_send(res, IBV_WR_RDMA_WRITE, &devx_info)) {
			loge("failed to post SR in producer thread\n");
			return NULL;
		}

		bf_base = devx_info.bf - EVEN_BF_OFFSET;
		ctrl_base = devx_info.ctrl;
		logi("bf_base = %p, ctrl_base = %p", bf_base, ctrl_base);

		// 如果是GPU模式，转换 bf 与 ctrl 指针到GPU
		if (config.mode == 1) {
			if (convert_blueflame_to_gpu(bf_base, ctrl_base) != 0) {
				loge("Failed to convert devx_info to GPU");
				return NULL;
			}
			logi("Successfully converted first devx_info to GPU");
		}

		void *current_bf = bf_base + EVEN_BF_OFFSET;
		void *current_ctrl = ctrl_base + 0x0;
		*((volatile uint64_t *)current_bf) = *(uint64_t *)current_ctrl;
		if (poll_completion(res)) {
			loge("even request: poll completion failed");
			return NULL;
		}

		// 产生 odd 任务
		if (post_send(res, IBV_WR_RDMA_WRITE, &devx_info)) {
			loge("failed to post SR in producer thread\n");
			return NULL;
		}

		current_bf = bf_base + ODD_BF_OFFSET;
		current_ctrl = ctrl_base + 0x40;
		*((volatile uint64_t *)current_bf) = *(uint64_t *)current_ctrl;
		if (poll_completion(res)) {
			loge("odd request: poll completion failed");
			return NULL;
		}
	}

	// 正式开始
	unsigned long perf_start_time_usec;
	unsigned long perf_cur_time_usec;
	struct timeval perf_cur_time;
	gettimeofday(&perf_cur_time, NULL);
	perf_start_time_usec = (perf_cur_time.tv_sec * 1000 * 1000) + (perf_cur_time.tv_usec);
	for (int i = 0; i < config.repeat && thread_running; ++i) {
		if (config.server_name) { // client
#if DEBUG_ENABLE_PRINT_DATA
			uint8_t value = i % 10;
			memset(res->buf, value, config.msg_size);
#endif
			struct ibv_devx_info devx_info;
			
			// 生产者线程控制逻辑：
			// 1. 检查队列是否有空间：request_cnt - complete_cnt < MAX_DEVX_INFO_COUNT
			// 2. 如果队列满，必须轮询完成事件来释放空间
			// 3. 如果complete_cnt < trigger_cnt，说明消费者没有触发doorbell，需要等待
			unsigned long start_time_msec;
			unsigned long cur_time_msec;
			struct timeval cur_time;
			gettimeofday(&cur_time, NULL);
			start_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
			while (thread_running) {
				// 首先轮询完成队列，更新complete_cnt

				if (request_cnt - complete_cnt >= (MAX_POLL_COUNT / 2)) {
					uint32_t completed = 0;
					uint32_t need_count = request_cnt - complete_cnt;
					need_count = need_count > MAX_POLL_COUNT ? MAX_POLL_COUNT : need_count;
					if (poll_completion_once(res, &completed, need_count) == 0 && completed > 0) {
						complete_cnt += completed;
						logi("Producer: poll_completion_once found %d completed tasks, cnt: (R %lu, T %lu, C %lu)", 
							completed, request_cnt, trigger_cnt, complete_cnt);
					}
				}
				
				// 检查队列是否有空间
				if ((request_cnt - complete_cnt) < MAX_DEVX_INFO_COUNT) {				
#if 0
					// 队列有空间的情况
					if (complete_cnt >= trigger_cnt) {
						// 情况a：队列有空间，但所有已触发doorbell的任务都已完成
						// 需要等待消费者继续触发doorbell
						// logi("Producer: queue has space but all triggered tasks completed, waiting for consumer, cnt: (R %lu, T %lu, C %lu)", 
							 request_cnt, trigger_cnt, complete_cnt);
						usleep(1000); // 1毫秒
					} else {
						// 情况b：队列有空间，且还有已触发doorbell但未完成的任务
						// 可以继续post_send请求新任务
						logi("Producer: queue has space and tasks in progress, can post new request, cnt: (R %lu, T %lu, C %lu)", 
							 request_cnt, trigger_cnt, complete_cnt);
					}
#endif
					break;
				} else {
					// 队列无空间的情况
					if (complete_cnt >= trigger_cnt) {
						// 情况a：消费者没有触发doorbell，任务无法完成
						// logi("Producer: queue full and consumer not triggering doorbell, waiting for consumer, cnt: (R %lu, T %lu, C %lu)", 
						// 	 request_cnt, trigger_cnt, complete_cnt);
						// usleep(1000); // 1毫秒
					} else {
						// 情况b：消费者已触发doorbell，但poll_completion没有轮询到完成事件
						// 这可能是网络延迟或硬件问题，继续轮询
						// logi("Producer: queue full but tasks triggered, waiting for completion, cnt: (R %lu, T %lu, C %lu)", 
						// 	 request_cnt, trigger_cnt, complete_cnt);
						// usleep(1000); // 1毫秒
					}
				}
				
				// 检查是否超时
				gettimeofday(&cur_time, NULL);
				cur_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
				if ((cur_time_msec - start_time_msec) >= MAX_POLL_CQ_TIMEOUT) {
					// 根据具体情况给出不同的超时错误信息
					if ((request_cnt - complete_cnt) >= MAX_DEVX_INFO_COUNT) {
						// 队列满的情况
						if (complete_cnt >= trigger_cnt) {
							loge("Producer timeout: queue full and consumer not triggering doorbell for %d ms", MAX_POLL_CQ_TIMEOUT);
						} else {
							loge("Producer timeout: queue full and no completion events for %d ms (possible hardware issue)", MAX_POLL_CQ_TIMEOUT);
						}
					} else {
						// 队列有空间但等待超时的情况
						loge("Producer timeout: waiting for consumer to trigger doorbell for %d ms", MAX_POLL_CQ_TIMEOUT);
					}
					break;
				}
			}
				
			if (!thread_running) {
				break;
			}

			if (post_send(res, IBV_WR_RDMA_WRITE, &devx_info)) {
				loge("failed to post SR in producer thread");
				break;
			}
			
			// post_send 成功后，先更新任务计数，然后计算索引
			request_cnt++;
			
			// 将 ctrl offset 放入队列
			uint32_t ctrl_offset = (uint32_t)(devx_info.ctrl - ctrl_base);
			ctrl_offsets[(request_cnt - 1) % MAX_DEVX_INFO_COUNT] = ctrl_offset;
			// WR idx 最大数为 (MAX_WR_COUNT * 4 - 1)，同时 ctrl 的长度为 MAX_WR_COUNT * 4 * 64
			logi("Producer: WR idx = %d, bf = %p, ctrl = %p, ctrl_offset = 0x%x, cnt: (R %lu, T %lu, C %lu)", 
					devx_info.idx, devx_info.bf, devx_info.ctrl, ctrl_offset, request_cnt, trigger_cnt, complete_cnt);
		}
	}
	
	// 等待所有任务完成，添加超时机制
	{
		unsigned long start_time_msec;
		unsigned long cur_time_msec;
		struct timeval cur_time;
		gettimeofday(&cur_time, NULL);
		start_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
		while (complete_cnt < request_cnt) {
			uint32_t completed = 0;
			uint32_t need_count = request_cnt - complete_cnt;
			need_count = need_count > MAX_POLL_COUNT ? MAX_POLL_COUNT : need_count;
			if (poll_completion_once(res, &completed, need_count) == 0 && completed > 0) {
				complete_cnt += completed;
				logi("Producer: poll_completion_once found %d completed tasks, cnt: (R %lu, T %lu, C %lu)", 
						completed, request_cnt, trigger_cnt, complete_cnt);

				// 获取到了完成任务，重置开始时间
				gettimeofday(&cur_time, NULL);
				start_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
			}
			
			// 检查超时
			gettimeofday(&cur_time, NULL);
			cur_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
			if ((cur_time_msec - start_time_msec) >= MAX_POLL_CQ_TIMEOUT) {
				loge("Producer: timeout waiting for completion, waited %lu ms, completed %lu/%lu tasks", 
					cur_time_msec - start_time_msec, complete_cnt, request_cnt);
				break;
			}
		}
	}
	gettimeofday(&perf_cur_time, NULL);
	perf_cur_time_usec = (perf_cur_time.tv_sec * 1000 * 1000) + (perf_cur_time.tv_usec);
	unsigned long perf_duration_usec = perf_cur_time_usec - perf_start_time_usec;
	
	// 计算吞吐量和平均延迟
	double throughput_mbps = 0.0;
	double avg_latency_us = 0.0;  // 改为double类型，避免精度丢失
	
	if (complete_cnt > 0 && perf_duration_usec > 0) {
		// 先计算平均延迟（微秒）
		avg_latency_us = (double)perf_duration_usec / (double)complete_cnt;
		// 带宽 = 消息大小 / 平均延迟 (转换为 MB/s)
		// 注意：这里需要将微秒转换为秒，所以乘以1000000
		throughput_mbps = (double)config.msg_size / avg_latency_us * 1000000.0 / (1024.0 * 1024.0);
	}

	logi("Producer thread finished, cnt: (R %lu, T %lu, C %lu)", request_cnt, trigger_cnt, complete_cnt);
	printf("size = %zu, complete %lu tasks, duration %lu us, throughput %.2f MB/s, avg latency %.2f us\n", 
		config.msg_size, complete_cnt, perf_duration_usec, throughput_mbps, avg_latency_us);

	return NULL;
}

static void* consumer_thread(void *arg)
{
	// struct resources *res = (struct resources *)arg;
	uint32_t ctrl_offset = 0;
	
	while (trigger_cnt < config.repeat && thread_running) {
		// 消费者线程控制逻辑：
		// 只能在 trigger_cnt < request_cnt 时才能触发门铃
		unsigned long start_time_msec;
		unsigned long cur_time_msec;
		struct timeval cur_time;
		
		gettimeofday(&cur_time, NULL);
		start_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
		// 等待生产者请求新任务，消费者线程好触发 doorbell，如果等待超时，则退出
		while (trigger_cnt >= request_cnt && thread_running) {
			// 检查是否超时
			gettimeofday(&cur_time, NULL);
			cur_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
			if ((cur_time_msec - start_time_msec) >= MAX_POLL_CQ_TIMEOUT) {
				loge("Consumer timeout: no new requests for %d ms", MAX_POLL_CQ_TIMEOUT);
				break;
			}
			// 短暂休眠避免过度占用CPU
			// usleep(1000); // 1毫秒
		}
		
		if (!thread_running) {
			break;
		}
		
		// 获取当前要处理的任务的ctrl_offset
		ctrl_offset = ctrl_offsets[trigger_cnt % MAX_DEVX_INFO_COUNT];
		
		// 根据模式选择触发方式
		if (config.mode == 1 && trigger_doorbell_func && gpu_bf && gpu_ctrl) {
			// GPU模式：使用GPU kernel触发门铃
			if (trigger_doorbell_func(gpu_bf, gpu_ctrl, ctrl_offset) != 0) {
				loge("GPU doorbell trigger failed, falling back to CPU mode");
				return NULL;
			}
			// 触发门铃后立即更新计数
			trigger_cnt++;
			logi("Consumer: gpu mode, trigger doorbell, ctrl_offset = 0x%x, cnt: (R %lu, T %lu, C %lu)", 
				ctrl_offset, request_cnt, trigger_cnt, complete_cnt);
		} else {
			// CPU模式：直接触发门铃
			uint64_t bf_offset = trigger_cnt % 2 ? ODD_BF_OFFSET : EVEN_BF_OFFSET;
			*((volatile uint64_t *)(bf_base + bf_offset)) = *(uint64_t *)(ctrl_base + ctrl_offset);
			// 触发门铃后立即更新计数
			trigger_cnt++;
			logi("Consumer: cpu mode, trigger doorbell, bf = %p, ctrl = %p, ctrl_offset = 0x%x, cnt: (R %lu, T %lu, C %lu)", 
				bf_base + bf_offset, ctrl_base + ctrl_offset, ctrl_offset, request_cnt, trigger_cnt, complete_cnt);
		}

	}
	
	logi("Consumer thread finished, cnt: (R %lu, T %lu, C %lu)", request_cnt, trigger_cnt, complete_cnt);
	return NULL;
}

// 只有当 MAX_WR_COUNT = 1时，才有意义，因为 buffer 只有一个，内容被覆盖，基本只打印最后一次任务的内容
static void* monitor_thread(void *arg)
{
#if !DEBUG_ENABLE_PRINT_DATA
	return NULL;
#endif

	struct resources *res = (struct resources *)arg;
	
	logi("Monitor thread started, monitoring res.buf[0]");
	uint8_t *p = (uint8_t *)res->buf;
			memset(res->buf, 0, config.msg_size);

	while (monitor_running) {
		printf("%d", *p);
		usleep(100);
	}
	
	printf("\n");
	loge("Monitor thread finished");
	return NULL;
}

int main(int argc, char *argv[])
{
	struct resources res;
	int rc = 1;
	char temp_char;
	/* parse the command line parameters */
	while (1)
	{
		int c;
		static struct option long_options[] = {
			{.name = "port", .has_arg = 1, .val = 'p'},
			{.name = "ib-dev", .has_arg = 1, .val = 'd'},
			{.name = "ib-port", .has_arg = 1, .val = 'i'},
			{.name = "gid-idx", .has_arg = 1, .val = 'g'},
			{.name = "repeat", .has_arg = 1, .val = 'r'},
			{.name = "mode", .has_arg = 1, .val = 'm'},
			{.name = "cuda-plugin", .has_arg = 1, .val = 'c'},
			{.name = "size", .has_arg = 1, .val = 's'},
			{.name = NULL, .has_arg = 0, .val = '\0'}
        };
		c = getopt_long(argc, argv, "p:d:i:g:r:m:c:s:", long_options, NULL);
		if (c == -1)
			break;
		switch (c)
		{
		case 'p':
			config.tcp_port = strtoul(optarg, NULL, 0);
			break;
		case 'd':
			config.dev_name = strdup(optarg);
			break;
		case 'i':
			config.ib_port = strtoul(optarg, NULL, 0);
			if (config.ib_port < 0)
			{
				usage(argv[0]);
				return 1;
			}
			break;
		case 'g':
			config.gid_idx = strtoul(optarg, NULL, 0);
			if (config.gid_idx < 0)
			{
				usage(argv[0]);
				return 1;
			}
			break;
		case 'r':
			config.repeat = strtoul(optarg, NULL, 0);
			if (config.repeat <= 0)
			{
				usage(argv[0]);
				return 1;
			}
			break;
		case 'm':
			config.mode = strtoul(optarg, NULL, 0);
			if (config.mode < 0 || config.mode > 1)
			{
				loge("Invalid mode: %d. Use 0 for CPU mode or 1 for GPU mode", config.mode);
				usage(argv[0]);
				return 1;
			}
			break;
		case 'c':
			strncpy(config.cuda_plugin_path, optarg, sizeof(config.cuda_plugin_path) - 1);
			config.cuda_plugin_path[sizeof(config.cuda_plugin_path) - 1] = '\0';
			break;
		case 's':
			config.msg_size = strtoul(optarg, NULL, 0);
			if (config.msg_size <= 0) {
				loge("Invalid message size: %s", optarg);
				usage(argv[0]);
				return 1;
			}
			break;
		default:
			usage(argv[0]);
			return 1;
		}
	}
	/* parse the last parameter (if exists) as the server name */
	if (optind == argc - 1)
		config.server_name = argv[optind];
    if(config.server_name){
        	logi("servername=%s", config.server_name);
    }
    else if (optind < argc)
    {
        usage(argv[0]);
        return 1;
    }
	/* print the used parameters for info*/
	print_config();
	
	// 加载CUDA插件（如果需要）
	if (config.mode == 1) {
		if (load_cuda_plugin() != 0) {
			loge("Failed to load CUDA plugin");
			rc = 1;
			goto main_exit;
		}
	}
	
	/* init all of the resources, so cleanup will be easy */
	resources_init(&res);
	/* create resources before using them */
	if (resources_create(&res))
	{
		loge("failed to create resources");
		goto main_exit;
	}
	
	/* connect the QPs */
	if (connect_qp(&res))
	{
		loge("failed to connect QPs");
		goto main_exit;
	}
	
	/* Sync so we are sure server side has data ready before client tries to read it */
	if (sock_sync_data(res.sock, 1, res.sync, &temp_char)) /* just send a dummy char back and forth */
	{
		loge("sync error before RDMA ops");
		rc = 1;
		goto main_exit;
	}

#if MULTI_THREAD_TEST_USE_BF // 模拟真实场景
	pthread_t monitor_tid; // server use
	if (config.server_name) { // client
		// 启动生产者线程和消费者线程
		pthread_t producer_tid, consumer_tid;
		sleep(1);
		logi("Starting producer and consumer threads...");
		
		if (pthread_create(&producer_tid, NULL, producer_thread, &res) != 0) {
			loge("failed to create producer thread");
			rc = 1;
			goto main_exit;
		}
		
		if (pthread_create(&consumer_tid, NULL, consumer_thread, &res) != 0) {
			loge("failed to create consumer thread");
			thread_running = 0;
			pthread_join(producer_tid, NULL);
			rc = 1;
			goto main_exit;
		}
		
		// 等待线程完成
		pthread_join(producer_tid, NULL);
		pthread_join(consumer_tid, NULL);
		
		logi("All threads completed");
	} else {
		// Server端：启动监控线程持续打印res.buf[0]
		if (pthread_create(&monitor_tid, NULL, monitor_thread, &res) != 0) {
			loge("failed to create monitor thread");
			rc = 1;
			goto main_exit;
		}
	}

	/* Sync so server will know that client is done mucking with its memory */
	if (sock_sync_data(res.sock, 1, res.sync, &temp_char)) /* just send a dummy char back and forth */
	{
		loge("sync error after RDMA ops");
		rc = 1;
		goto main_exit;
	}

	if (!config.server_name) // server
	{
		monitor_running = 0;
		pthread_join(monitor_tid, NULL);

		// 打印最后一次的值
#if DEBUG_ENABLE_PRINT_DATA
		printf("server got data: ");
		for (int n = 0; n < config.msg_size; ++n)
		{
			printf("%d, ", (uint8_t)res.buf[n]);
		}
		printf("\n");
#endif
	}

	rc = 0;
main_exit:
	// 停止线程运行
	thread_running = 0;
	monitor_running = 0;
	
	// 卸载CUDA插件
	unload_cuda_plugin();
#else
	uint32_t test_sizes_arr[] = {1 << 2, 1 << 3, 1 << 4, 1 << 5, 
		1 << 6, 1 << 7, 1 << 8, 1 << 9, 1 << 10, 1 << 11, 
		1 << 12, 1 << 13, 1 << 14, 1 << 15, 1 << 16, 
		1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22, 1 << 23};

	printf("NORMAL_TEST_USE_BF = %d\n", NORMAL_TEST_USE_BF);
	if (config.server_name) {
		int warmup = 5;
		struct ibv_devx_info devx_info;
		size_t test_sizes_array_size = sizeof(test_sizes_arr) / sizeof(test_sizes_arr[0]);

		for (int l = 0; l < test_sizes_array_size; l++) {
			config.msg_size = test_sizes_arr[l];
			for (int i = 0; i < warmup; i++) {
				if (post_send(&res, IBV_WR_RDMA_WRITE, &devx_info))
				{
					loge("failed to post");
					rc = 1;
					goto main_exit;
				}

#if NORMAL_TEST_USE_BF
				// doorbell
				// logw("WR idx = %d, bf = %p, ctrl = %p", devx_info.idx, devx_info.bf, devx_info.ctrl);
				*((volatile uint64_t *)devx_info.bf) = *(uint64_t *)devx_info.ctrl; // 按门铃成功，mlx5 里的按门铃函数 mmio_write64_be 非常的复杂
#endif

				if (poll_completion(&res))
				{
					loge("poll completion failed");
					rc = 1;
					goto main_exit;
				}
			}

			// 正式开始性能测试
			unsigned long perf_start_time_usec;
			unsigned long perf_cur_time_usec;
			struct timeval perf_cur_time;
			gettimeofday(&perf_cur_time, NULL);
			perf_start_time_usec = (perf_cur_time.tv_sec * 1000 * 1000) + (perf_cur_time.tv_usec);
			for (int i = 0; i < config.repeat; i++) {
				if (post_send(&res, IBV_WR_RDMA_WRITE, &devx_info))
				{
					loge("failed to post");
					rc = 1;
					goto main_exit;
				}

#if NORMAL_TEST_USE_BF
				// doorbell
				// logw("WR idx = %d, bf = %p, ctrl = %p", devx_info.idx, devx_info.bf, devx_info.ctrl);
				*((volatile uint64_t *)devx_info.bf) = *(uint64_t *)devx_info.ctrl; // 按门铃成功，mlx5 里的按门铃函数 mmio_write64_be 非常的复杂
#endif
				if (poll_completion(&res))
				{
					loge("poll completion failed");
					rc = 1;
					goto main_exit;
				}
			}
			gettimeofday(&perf_cur_time, NULL);
			perf_cur_time_usec = (perf_cur_time.tv_sec * 1000 * 1000) + (perf_cur_time.tv_usec);
			unsigned long perf_duration_usec = perf_cur_time_usec - perf_start_time_usec;
			
			// 计算吞吐量和平均延迟
			double throughput_mbps = 0.0;
			double avg_latency_us = 0.0;  // 改为double类型，避免精度丢失
			
			avg_latency_us = (double)perf_duration_usec / (double)config.repeat;
			// 带宽 = 消息大小 / 平均延迟 (转换为 MB/s)
			// 注意：这里需要将微秒转换为秒，所以乘以1000000
			throughput_mbps = (double)config.msg_size / avg_latency_us * 1000000.0 / (1024.0 * 1024.0);

			printf("size = %zu, complete %d tasks, duration %lu us, throughput %.2f MB/s, avg latency %.2f us\n", 
				config.msg_size, config.repeat, perf_duration_usec, throughput_mbps, avg_latency_us);
		}
	}

	/* Sync so server will know that client is done mucking with its memory */
	if (sock_sync_data(res.sock, 1, res.sync, &temp_char)) /* just send a dummy char back and forth */
	{
		loge("sync error after RDMA ops");
		rc = 1;
		goto main_exit;
	}

	printf("test success!\n");

main_exit:	

#endif
	
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
