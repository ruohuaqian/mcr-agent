from torch.utils.data import IterableDataset, get_worker_info, DataLoader

class BatchStreamDataset(IterableDataset):
    def __init__(self, task_iter_fn, batch_size, model_ref):
        """
        task_iter_fn: 可调用，返回一个生成器/迭代器，逐个产出 {task, repeat_idx, swapColor, ...}
        batch_size:   目标批大小
        model_ref:    持有 preprocess_function / collate_once 的对象（通常就是你的 model）
        """
        self.task_iter_fn = task_iter_fn
        self.batch_size = batch_size
        self.M = model_ref

    def _sample_stream(self):
        # 由你提供：从某个索引文件 / 列表 / API 源源不断 yield 样本描述
        yield from self.task_iter_fn()

    def __iter__(self):
        wi = get_worker_info()
        # —— 关键：对“样本序号”做 worker 分片，避免并行重复 —— #
        if wi is None:
            enumerated = enumerate(self._sample_stream())
            shard_pred = lambda i: True
        else:
            enumerated = enumerate(self._sample_stream())
            shard_pred = lambda i: (i % wi.num_workers) == wi.id

        batch_buf = []
        meta_buf  = []

        for i, sample_desc in enumerated:
            if not shard_pred(i):
                continue

            # 单样本：下载+解析+特征构造（不落盘、用 HfFileSystem，在你的 preprocess_function 里处理）
            ex = self.M.preprocess_function(sample_desc)   # dict processed_data（含 raw_ex）
            batch_buf.append(ex)
            # raw_ex 会在 collate_once 里被抽到 metadata，这里可不单存

            if len(batch_buf) == self.batch_size:
                # 批次：仅做对齐/打包，不做 IO
                feat, meta = self.M.collate_once(batch_buf) # 等价你 collate_fn(list_of_examples, device)
                yield (feat, meta)
                batch_buf.clear()

        # 尾批（不足 BATCH_SIZE）
        if batch_buf:
            feat, meta = self.M.collate_once(batch_buf)
            yield (feat, meta)

# —— 你的“只做对齐/打包”的函数（把之前给你的 collate_fn 稍微改名即可）——
def collate_once(self, list_of_examples):
    device = torch.device('cuda' if self.args.gpu else 'cpu')
    return self.collate_fn(list_of_examples, device)  # 直接复用你已经实现的 collate_fn
