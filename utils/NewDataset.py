class PairConcatARCDataset(ARCDataset):
    """
    将同一个任务的样本两两配对，进行垂直拼接
    每个返回的样本 = sample_i + sample_j (垂直拼接)
    """
    
    def __init__(
        self,
        root: Path,
        split: str,
        subset: str = "train",
        max_size: int = 32,
        task_lookup: Optional[Dict[str, int]] = None,
        extra_train_roots: Optional[Iterable[Path]] = None,
        extra_train_limit: Optional[int] = None,
    ) -> None:
        # 先调用父类初始化，加载所有样本
        super().__init__(
            root=root,
            split=split,
            subset=subset,
            max_size=max_size,
            task_lookup=task_lookup,
            extra_train_roots=extra_train_roots,
            extra_train_limit=extra_train_limit,
        )
        
        # 重新组织样本：按任务分组，然后两两配对
        self.paired_samples = self._create_sample_pairs()
        
    def _create_sample_pairs(self) -> List[Dict]:
        """将样本按任务分组，每两个样本配成一对"""
        from collections import defaultdict
        
        # 按任务分组
        task_groups = defaultdict(list)
        for sample in self.samples:
            task_id = sample["task_index"]
            task_groups[task_id].append(sample)
        
        paired_samples = []
        
        for task_id, task_samples in task_groups.items():
            # 对每个任务的样本进行两两配对
            for i in range(0, len(task_samples) - 1, 2):  # 步长为2
                sample1 = task_samples[i]
                sample2 = task_samples[i + 1]
                
                # 创建配对样本
                paired_sample = {
                    "sample1": sample1,
                    "sample2": sample2,
                    "task_index": task_id,
                    "task_name": sample1["task_name"],  # 保持任务名
                }
                paired_samples.append(paired_sample)
        
        print(f"Created {len(paired_samples)} paired samples from {len(self.samples)} original samples")
        return paired_samples
    
    def __len__(self) -> int:
        return len(self.paired_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        paired_sample = self.paired_samples[idx]
        
        # 获取两个样本的处理结果
        sample1_data = self._process_single_sample(paired_sample["sample1"])
        sample2_data = self._process_single_sample(paired_sample["sample2"])
        
        # 垂直拼接两个样本
        input1 = sample1_data["inputs"]      # (max_size, max_size)
        input2 = sample2_data["inputs"]      # (max_size, max_size)
        mask1 = sample1_data["attention_mask"]
        mask2 = sample2_data["attention_mask"]
        
        # 垂直拼接
        combined_input = torch.cat([input1, input2], dim=0)    # (max_size*2, max_size)
        combined_mask = torch.cat([mask1, mask2], dim=0)      # (max_size*2, max_size)
        
        # 合并 targets（如果需要的话）
        target1 = sample1_data["targets"]
        target2 = sample2_data["targets"]
        combined_target = torch.cat([target1, target2], dim=0)
        
        return {
            "inputs": combined_input,           # (128, 64) - 垂直拼接后的输入
            "attention_mask": combined_mask,    # (128, 64) - 对应的掩码
            "targets": combined_target,         # (128, 64) - 拼接后的目标
            "task_id": torch.tensor(paired_sample["task_index"], dtype=torch.long),
            "task_name": paired_sample["task_name"],
            # 可选：保留原始样本信息用于调试
            "sample1_data": sample1_data,
            "sample2_data": sample2_data,
        }
    
    def _process_single_sample(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """处理单个样本，返回标准格式的数据"""
        # 这里需要调用父类的样本处理逻辑
        # 但是由于 __getitem__ 是私有方法，我们需要复制处理逻辑
        
        example = sample["example"]
        task_index = sample["task_index"]
        task_name = sample["task_name"]
        example_index = sample["example_index"]
        
        # 调用父类的 _process_example 方法
        processed = self._process_example(
            example=example,
            task_index=task_index,
            task_name=task_name,
            example_index=example_index,
            if_translation=self.translation_enabled,
            rng=self.rng,
        )
        
        return processed