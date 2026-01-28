# π0.5 代码走读指南

## 代码库信息
- **GitHub**: https://github.com/Physical-Intelligence/openpi
- **框架**: JAX + Flax
- **语言**: Python 3.10+
- **依赖**: PaliGemma, FAST tokenizer, Flow Matching

---

## 目录结构

根据论文描述和官方文档，预期的代码结构：

```
openpi/
├── openpi/
│   ├── training/
│   │   ├── config.py              # 训练配置
│   │   ├── train.py               # 训练主循环
│   │   └── losses.py              # 损失函数
│   ├── policies/
│   │   ├── policy_config.py       # 策略配置
│   │   ├── pi05.py                # π0.5模型定义
│   │   ├── pi0.py                 # π0模型定义
│   │   └── pi0_fast.py            # π0-FAST模型
│   ├── models/
│   │   ├── vla.py                 # VLA基础架构
│   │   ├── action_expert.py       # Action Expert
│   │   ├── attention.py           # 注意力机制
│   │   └── flow_matching.py       # Flow Matching实现
│   ├── data/
│   │   ├── dataset.py             # 数据集加载
│   │   ├── transforms.py          # 数据增强
│   │   └── tokenizers/
│   │       ├── fast.py            # FAST tokenizer
│   │       └── text.py            # 文本tokenizer
│   ├── shared/
│   │   ├── download.py            # 模型下载
│   │   └── utils.py               # 工具函数
│   └── evaluation/
│       ├── eval.py                # 评估脚本
│       └── metrics.py             # 评估指标
├── scripts/
│   ├── train.py                   # 训练入口
│   ├── eval.py                    # 评估入口
│   └── inference.py               # 推理入口
├── configs/
│   ├── pi05_base.yaml
│   ├── pi05_droid.yaml
│   └── pi05_libero.yaml
└── README.md
```

---

## 核心模块详解

### 1. 模型定义 (`openpi/policies/pi05.py`)

#### 1.1 主模型类

**对应论文**: Section IV.A (π0.5 Architecture)

```python
class Pi05Policy(nn.Module):
    """π0.5 Vision-Language-Action Policy
    
    对应论文公式:
    πθ(at:t+H, ˆℓ | ot, ℓ) = πθ(at:t+H | ot, ˆℓ) · πθ(ˆℓ | ot, ℓ)
    """
    
    def __init__(
        self,
        vlm_backbone: str = "paligemma-2b",     # VLM backbone
        action_expert_dim: int = 1024,           # Action expert维度
        action_horizon: int = 50,                # 动作chunk长度
        action_dim: int = 19,                    # 动作维度
        use_fast_tokens: bool = True,            # 是否使用FAST
        flow_matching_steps: int = 10,           # 去噪步数
    ):
        super().__init__()
        
        # 1. VLM Backbone (SigLIP + Gemma)
        # 论文: "builds upon π0 and adopts the PaliGemma VLM"
        self.vlm = PaliGemmaModel.from_pretrained(vlm_backbone)
        self.vlm_dim = 2048  # Gemma-2B的隐藏维度
        
        # 2. Action Expert
        # 论文 Section IV.A: "action expert, as with π0"
        self.action_expert = ActionExpert(
            input_dim=action_dim,
            hidden_dim=action_expert_dim,
            num_layers=18,
            num_heads=8,
        )
        
        # 3. Timestep Embedding (for flow matching)
        # 论文 Appendix E: "uses a separate MLP for projecting τ"
        self.timestep_mlp = TimestepMLP(
            input_dim=256,  # Sinusoidal encoding dim
            hidden_dim=action_expert_dim
        )
        
        # 4. FAST Tokenizer (optional, for pre-training)
        if use_fast_tokens:
            self.fast_tokenizer = FASTTokenizer(
                action_dim=action_dim,
                codebook_size=1024,
            )
        
        # 5. Action projection layers
        self.action_proj_in = nn.Dense(action_expert_dim)   # a → hidden
        self.action_proj_out = nn.Dense(action_dim)         # hidden → a
        
    def __call__(
        self,
        images: jnp.ndarray,           # [B, N_cam, H, W, 3]
        language: jnp.ndarray,         # [B, N_text]
        proprio: jnp.ndarray,          # [B, D_proprio]
        actions: Optional[jnp.ndarray] = None,  # [B, H, D_action]
        tau: Optional[float] = None,   # Flow matching timestep
        mode: str = "high_level",      # "high_level" or "low_level"
    ):
        """
        前向传播
        
        Args:
            images: 多相机图像
            language: 语言指令token
            proprio: 本体感知状态
            actions: 动作序列 (训练时提供)
            tau: Flow matching时间步 (训练时提供)
            mode: 推理模式
            
        Returns:
            如果mode="high_level": 输出子任务token
            如果mode="low_level": 输出动作序列
        """
        
        # Step 1: VLM encoding
        # 论文: "The VLM backbone takes in a sequence of images and a language prompt"
        vlm_output = self.encode_with_vlm(images, language, proprio)
        
        # Step 2: 根据模式选择推理路径
        if mode == "high_level":
            # 高层推理: 预测子任务
            return self.high_level_inference(vlm_output)
        else:
            # 低层推理: 预测动作
            return self.low_level_inference(vlm_output, actions, tau)
            
    def encode_with_vlm(self, images, language, proprio):
        """VLM编码
        
        对应论文 Section IV.A:
        "The model corresponds to a transformer that takes in N multimodal 
         input tokens x1:N"
        """
        
        # 1. Image encoding via SigLIP
        # 论文: "image patches are fed through a vision encoder"
        B, N_cam = images.shape[:2]
        image_features = []
        for i in range(N_cam):
            feat = self.vlm.vision_encoder(images[:, i])  # [B, 256, D]
            image_features.append(feat)
        image_tokens = jnp.concatenate(image_features, axis=1)  # [B, 1024, D]
        
        # 2. Text tokenization
        # 论文: "text tokens are embedded with an embedding matrix"
        text_tokens = self.vlm.text_encoder(language)  # [B, N_text, D]
        
        # 3. Proprioception tokenization
        # 论文: "The robot proprioceptive state is discretized"
        proprio_discrete = self.discretize_proprio(proprio)
        proprio_tokens = self.vlm.text_encoder.embed(proprio_discrete)  # [B, 1, D]
        
        # 4. Concatenate all tokens
        all_tokens = jnp.concatenate([
            image_tokens,      # 1024 tokens (4 cameras × 256)
            text_tokens,       # ~20 tokens
            proprio_tokens,    # 1 token
        ], axis=1)
        
        # 5. Apply VLM transformer
        # 注意: 这里使用的是双向注意力（对于prefix部分）
        vlm_encoded = self.vlm.transformer(
            all_tokens,
            attention_mask=self.create_prefix_mask(all_tokens)
        )
        
        return vlm_encoded
        
    def high_level_inference(self, vlm_output):
        """高层推理: 预测子任务
        
        对应论文公式:
        πθ(ˆℓ | ot, ℓ)
        """
        
        # 自回归生成子任务token
        # 论文: "the model first produces a high-level subtask"
        logits = self.vlm.lm_head(vlm_output)
        
        return logits  # [B, Vocab_size]
        
    def low_level_inference(self, vlm_output, actions=None, tau=None):
        """低层推理: 预测动作
        
        对应论文公式:
        πθ(at:t+H | ot, ˆℓ)
        
        使用Flow Matching:
        学习向量场 v = ω - at:t+H
        """
        
        if self.training and actions is not None:
            # 训练模式: Flow Matching
            return self.flow_matching_train(vlm_output, actions, tau)
        else:
            # 推理模式: 迭代去噪
            return self.flow_matching_inference(vlm_output)
            
    def flow_matching_train(self, vlm_output, actions, tau):
        """Flow Matching训练
        
        对应论文 Section IV.B:
        "Given aτ,ω = τa + (1-τ)ω, the model is trained to predict ω - a"
        """
        
        # 1. 采样噪声
        B, H, D = actions.shape
        noise = jax.random.normal(self.make_rng(), shape=(B, H, D))
        
        # 2. 生成带噪声的动作
        # 论文公式: aτ,ω = τ·a + (1-τ)·ω
        noisy_actions = tau * actions + (1 - tau) * noise
        
        # 3. 投影到隐藏空间
        action_hidden = self.action_proj_in(noisy_actions)  # [B, H, D_hidden]
        
        # 4. 时间步编码
        # 论文 Appendix E: "uses adaptive RMSNorm to inject timestep information"
        timestep_emb = self.timestep_mlp(
            self.sinusoidal_encoding(tau)
        )  # [B, D_hidden]
        
        # 5. Action Expert前向
        # 论文: "These tokens also use a different set of model weights"
        action_output = self.action_expert(
            action_hidden,
            vlm_context=vlm_output,
            timestep_emb=timestep_emb,
            attention_mask=self.create_action_mask(action_hidden, vlm_output)
        )
        
        # 6. 预测向量场
        # 论文: "output the flow matching vector field"
        pred_velocity = self.action_proj_out(action_output)  # [B, H, D]
        
        # 7. 目标: ω - a
        target_velocity = noise - actions
        
        return pred_velocity, target_velocity
        
    def flow_matching_inference(self, vlm_output, num_steps=10):
        """Flow Matching推理
        
        对应论文: "10 denoising steps"
        """
        
        B = vlm_output.shape[0]
        H = self.action_horizon
        D = self.action_dim
        
        # 1. 初始化: 纯噪声
        actions = jax.random.normal(self.make_rng(), shape=(B, H, D))
        
        # 2. 迭代去噪
        # 论文: "iterative integration of the flow field"
        delta_tau = 1.0 / num_steps
        for i in range(num_steps):
            tau = i * delta_tau
            
            # 2.1 编码时间步
            timestep_emb = self.timestep_mlp(
                self.sinusoidal_encoding(tau)
            )
            
            # 2.2 投影动作
            action_hidden = self.action_proj_in(actions)
            
            # 2.3 Action Expert预测向量场
            action_output = self.action_expert(
                action_hidden,
                vlm_context=vlm_output,
                timestep_emb=timestep_emb,
            )
            
            # 2.4 更新动作
            velocity = self.action_proj_out(action_output)
            actions = actions + delta_tau * velocity
            
        return actions  # [B, H, D] - 干净的动作序列
        
    def create_prefix_mask(self, tokens):
        """创建prefix的注意力mask
        
        对应论文 Appendix E 的 Figure 18:
        图像、文本、本体感知使用双向注意力
        """
        N = tokens.shape[1]
        # 所有prefix token可以相互注意
        mask = jnp.ones((N, N))
        return mask
        
    def create_action_mask(self, action_tokens, vlm_tokens):
        """创建Action Expert的注意力mask
        
        对应论文:
        "Action tokens attend to the prefix and to one another"
        "do not attend to FAST action tokens"
        """
        N_vlm = vlm_tokens.shape[1]
        N_act = action_tokens.shape[1]
        
        mask = jnp.zeros((N_act, N_vlm + N_act))
        
        # 1. Action tokens可以看到所有VLM tokens
        mask[:, :N_vlm] = 1
        
        # 2. Action tokens可以相互看到（双向注意力）
        mask[:, N_vlm:] = 1
        
        return mask
        
    @staticmethod
    def sinusoidal_encoding(t, dim=256):
        """时间步的正弦编码
        
        对应论文 Appendix E:
        "ϕ : ℝ → ℝʷ is a sinusoidal positional encoding function"
        """
        half_dim = dim // 2
        freqs = jnp.exp(-jnp.log(10000) * jnp.arange(half_dim) / half_dim)
        args = t[..., None] * freqs[None, :]
        encoding = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
        return encoding
```

#### 1.2 Action Expert实现

**对应论文**: Section IV.A, Appendix E

```python
class ActionExpert(nn.Module):
    """Action Expert Transformer
    
    对应论文:
    "These tokens also use a different set of model weights, 
     which we refer to as an 'action expert'"
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 18,
        num_heads: int = 8,
        mlp_dim: int = 4096,
    ):
        super().__init__()
        
        # Transformer layers
        self.layers = [
            ActionExpertLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
            )
            for _ in range(num_layers)
        ]
        
    def __call__(
        self,
        action_hidden,          # [B, H, D]
        vlm_context,            # [B, N_vlm, D]
        timestep_emb,           # [B, D]
        attention_mask=None,
    ):
        """
        Args:
            action_hidden: 投影后的动作表示
            vlm_context: VLM的输出
            timestep_emb: 时间步嵌入
            attention_mask: 注意力mask
        """
        
        x = action_hidden
        
        # 逐层处理
        for layer in self.layers:
            x = layer(
                x,
                context=vlm_context,
                timestep_emb=timestep_emb,
                mask=attention_mask
            )
            
        return x


class ActionExpertLayer(nn.Module):
    """Action Expert的单层
    
    对应论文 Appendix E:
    "applies adaptive RMSNorm to inject the timestep information"
    """
    
    def __init__(self, hidden_dim, num_heads, mlp_dim):
        super().__init__()
        
        # 1. Self-attention (within actions)
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads)
        self.self_attn_norm = AdaptiveRMSNorm(hidden_dim)
        
        # 2. Cross-attention (to VLM context)
        self.cross_attn = MultiHeadAttention(hidden_dim, num_heads)
        self.cross_attn_norm = AdaptiveRMSNorm(hidden_dim)
        
        # 3. MLP
        self.mlp = MLP(hidden_dim, mlp_dim)
        self.mlp_norm = AdaptiveRMSNorm(hidden_dim)
        
    def __call__(self, x, context, timestep_emb, mask=None):
        """
        Args:
            x: Action tokens [B, H, D]
            context: VLM context [B, N, D]
            timestep_emb: 时间步嵌入 [B, D]
        """
        
        # 1. Self-attention
        # 论文: "Action tokens attend to one another"
        normed_x = self.self_attn_norm(x, timestep_emb)
        x = x + self.self_attn(normed_x, normed_x, normed_x)
        
        # 2. Cross-attention to VLM
        # 论文: "attend to the prefix"
        normed_x = self.cross_attn_norm(x, timestep_emb)
        x = x + self.cross_attn(
            normed_x,      # query: action
            context,       # key: VLM
            context,       # value: VLM
            mask=mask
        )
        
        # 3. MLP
        normed_x = self.mlp_norm(x, timestep_emb)
        x = x + self.mlp(normed_x)
        
        return x


class AdaptiveRMSNorm(nn.Module):
    """Adaptive RMSNorm with timestep conditioning
    
    对应论文 Appendix E:
    "uses a separate MLP for projecting τ only and then applies 
     adaptive RMSNorm to inject the timestep information"
    """
    
    def __init__(self, dim):
        super().__init__()
        self.scale = self.param('scale', nn.initializers.ones, (dim,))
        
    def __call__(self, x, timestep_emb):
        """
        Args:
            x: [B, N, D]
            timestep_emb: [B, D]
        """
        
        # 1. RMSNorm
        # RMS = sqrt(mean(x^2) + eps)
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
        normed = x / rms * self.scale
        
        # 2. Modulate with timestep
        # timestep_emb作为scale调制
        scale = 1.0 + timestep_emb[:, None, :]  # [B, 1, D]
        modulated = normed * scale
        
        return modulated
```

---

### 2. 训练流程 (`openpi/training/train.py`)

#### 2.1 两阶段训练

**对应论文**: Section IV.C & IV.D

```python
def train_pi05(config):
    """完整的π0.5训练流程"""
    
    # ========== Stage 1: Pre-training ==========
    # 论文 Section IV.C: "a pre-training stage intended to adapt the model 
    #                      to diverse robotic tasks"
    
    print("=" * 50)
    print("Stage 1: Pre-training (280k steps)")
    print("=" * 50)
    
    # 1.1 加载数据
    pretrain_datasets = load_pretrain_datasets(
        mm_data=True,      # Mobile manipulator
        me_data=True,      # Multi-environment  
        ce_data=True,      # Cross-embodiment
        hl_data=True,      # High-level subtasks
        wd_data=True,      # Web data
    )
    
    # 1.2 初始化模型（从PaliGemma）
    model = Pi05Policy.from_pretrained("paligemma-2b")
    
    # 1.3 配置optimizer
    optimizer = optax.adamw(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # 1.4 训练循环
    for step in range(280_000):
        # 采样batch
        batch = sample_batch(pretrain_datasets)
        
        # 计算loss（仅文本cross-entropy）
        # 论文 Equation (1) with α=0
        loss, grads = compute_pretrain_loss(
            model, batch, use_fast_tokens=True
        )
        
        # 更新参数
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        # 记录
        if step % 1000 == 0:
            log_metrics(step, loss)
            
    print("Pre-training完成!")
    save_checkpoint(model, "pi05_pretrained.ckpt")
    
    
    # ========== Stage 2: Post-training ==========
    # 论文 Section IV.D: "a post-training stage intended to specialize 
    #                      it to mobile manipulation"
    
    print("=" * 50)
    print("Stage 2: Post-training (80k steps)")
    print("=" * 50)
    
    # 2.1 加载预训练模型
    model = load_checkpoint("pi05_pretrained.ckpt")
    
    # 2.2 初始化Action Expert
    # 论文: "the action expert (which is initialized with random weights)"
    model.init_action_expert()
    
    # 2.3 加载post-training数据
    posttrain_datasets = load_posttrain_datasets(
        mm_data=True,      # Mobile manipulator (filtered)
        me_data=True,      # Multi-environment (filtered)  
        hl_data=True,      # High-level subtasks
        wd_data=True,      # Web data
        vi_data=True,      # Verbal instructions (NEW!)
    )
    
    # 2.4 训练循环
    for step in range(80_000):
        # 采样batch
        batch = sample_batch(posttrain_datasets)
        
        # 计算联合loss
        # 论文 Equation (1) with α=10.0
        loss, grads = compute_posttrain_loss(
            model,
            batch,
            alpha=10.0,  # Flow matching weight
        )
        
        # 更新参数
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        # 记录
        if step % 1000 == 0:
            log_metrics(step, loss)
            
    print("Post-training完成!")
    save_checkpoint(model, "pi05_final.ckpt")


def compute_pretrain_loss(model, batch, use_fast_tokens=True):
    """Pre-training loss
    
    对应论文 Equation (1) with α=0:
    L = H(x1:M, f^ℓ_θ(ot, ℓ))
    """
    
    images = batch['images']        # [B, N_cam, H, W, 3]
    language = batch['language']    # [B, N_text]
    proprio = batch['proprio']      # [B, D_proprio]
    actions = batch['actions']      # [B, H, D_action]
    
    # 1. VLM encoding
    vlm_output = model.encode_with_vlm(images, language, proprio)
    
    # 2. Action tokenization (FAST)
    if use_fast_tokens:
        action_tokens = model.fast_tokenizer.encode(actions)
        target_tokens = action_tokens
    else:
        # 如果是web data，target是text
        target_tokens = batch.get('target_text', action_tokens)
    
    # 3. Autoregressive prediction
    logits = model.vlm.lm_head(vlm_output)
    
    # 4. Cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, target_tokens
    )
    
    return loss.mean()


def compute_posttrain_loss(model, batch, alpha=10.0):
    """Post-training loss
    
    对应论文 Equation (1):
    L = H(x1:M, f^ℓ_θ) + α·||ω - a - f^a_θ||²
    """
    
    images = batch['images']
    language = batch['language']
    proprio = batch['proprio']
    actions = batch['actions']
    
    # 1. 采样flow matching timestep
    # 论文: "Beta(s-τ/s; α=1.5, β=1)"
    tau = sample_flow_timestep(
        batch_size=actions.shape[0],
        alpha=1.5,
        beta=1.0,
        threshold=0.999,
    )
    
    # 2. VLM encoding
    vlm_output = model.encode_with_vlm(images, language, proprio)
    
    # 3. Text loss (包括FAST tokens)
    logits = model.vlm.lm_head(vlm_output)
    text_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, batch['target_tokens']
    )
    
    # 4. Flow matching loss
    pred_velocity, target_velocity = model.flow_matching_train(
        vlm_output, actions, tau
    )
    flow_loss = jnp.mean((pred_velocity - target_velocity) ** 2)
    
    # 5. Combined loss
    # 论文: "α = 10.0"
    total_loss = text_loss.mean() + alpha * flow_loss
    
    return total_loss


def sample_flow_timestep(batch_size, alpha, beta, threshold):
    """采样flow matching timestep
    
    对应论文 Appendix E:
    "p(τ) = Beta(s-τ/s; α=1.5, β=1), s=0.999"
    """
    
    # Beta分布采样
    raw_tau = jax.random.beta(
        key,
        a=alpha,
        b=beta,
        shape=(batch_size,)
    )
    
    # 映射到[0, threshold]
    tau = (1 - raw_tau) * threshold
    
    return tau
```

---

### 3. 推理流程 (`scripts/inference.py`)

#### 3.1 层次化推理

**对应论文**: Section V, Figure 2

```python
class Pi05Agent:
    """π0.5 Agent with hierarchical inference"""
    
    def __init__(self, model_path, device="cuda"):
        # 加载模型
        self.model = load_model(model_path, device=device)
        self.model.eval()
        
        # 推理配置
        self.high_level_freq = 0.5  # Hz (每2秒)
        self.low_level_freq = 50.0  # Hz
        self.action_horizon = 50    # 1秒的动作
        
        # 状态
        self.current_subtask = None
        self.last_high_level_time = 0
        
    def run_episode(self, task_description, env, max_steps=1000):
        """执行一个完整的episode
        
        对应论文 Figure 2:
        "The robot is given general tasks (close the cabinets, ...), 
         which it performs by both predicting subtasks and emitting 
         low-level actions"
        """
        
        observation = env.reset()
        done = False
        step = 0
        
        while not done and step < max_steps:
            # 1. 高层推理（如果需要）
            # 论文: "the model first predicts the semantic subtask"
            if self.should_update_high_level():
                self.current_subtask = self.high_level_inference(
                    observation, task_description
                )
                print(f"[High-Level] Subtask: {self.current_subtask}")
            
            # 2. 低层推理
            # 论文: "predicts the low-level robot action chunk"
            action_chunk = self.low_level_inference(
                observation, self.current_subtask
            )
            
            # 3. 执行动作（action chunking）
            for t in range(self.action_horizon):
                observation, reward, done, info = env.step(action_chunk[t])
                step += 1
                
                # 实时更新观察
                if step % 10 == 0:  # 每200ms更新一次
                    break  # 重新推理
                    
        return step
        
    def should_update_high_level(self):
        """判断是否需要更新高层推理
        
        对应论文: "At runtime, during each step of inference"
        """
        current_time = time.time()
        elapsed = current_time - self.last_high_level_time
        
        if elapsed >= 1.0 / self.high_level_freq:
            self.last_high_level_time = current_time
            return True
        return False
        
    def high_level_inference(self, observation, task):
        """高层推理：预测子任务
        
        对应论文:
        "the model first predicts the semantic subtask, inferring 
         the behavior that is appropriate to perform next"
        """
        
        # 1. 准备输入
        images = self.preprocess_images(observation['images'])
        language = self.tokenize_text(task)
        proprio = observation['proprio']
        
        # 2. VLM encoding
        with torch.no_grad():
            vlm_output = self.model.encode_with_vlm(
                images, language, proprio
            )
        
        # 3. 自回归生成子任务
        # 论文: "high-level inference captures πθ(ˆℓ | ot, ℓ)"
        subtask_tokens = self.autoregressive_decode(
            vlm_output,
            max_length=20,
            temperature=0.7,
        )
        
        # 4. 解码为文本
        subtask_text = self.detokenize(subtask_tokens)
        
        return subtask_text
        
    def low_level_inference(self, observation, subtask):
        """低层推理：预测动作序列
        
        对应论文:
        "low-level inference captures πθ(at:t+H | ot, ˆℓ)"
        """
        
        # 1. 准备输入（现在language是subtask）
        images = self.preprocess_images(observation['images'])
        language = self.tokenize_text(subtask)
        proprio = observation['proprio']
        
        # 2. VLM encoding
        with torch.no_grad():
            vlm_output = self.model.encode_with_vlm(
                images, language, proprio
            )
        
        # 3. Flow matching inference（10步去噪）
        # 论文: "At inference time we then use ... 10 denoising steps"
        actions = self.model.flow_matching_inference(
            vlm_output,
            num_steps=10
        )
        
        return actions  # [H, D_action]
        
    def autoregressive_decode(self, context, max_length, temperature):
        """自回归解码
        
        用于生成子任务token序列
        """
        
        tokens = []
        for _ in range(max_length):
            # 预测下一个token
            logits = self.model.high_level_inference(context)
            
            # Temperature sampling
            probs = jax.nn.softmax(logits / temperature)
            next_token = jax.random.categorical(self.rng, probs)
            
            # 检查是否结束
            if next_token == self.eos_token:
                break
                
            tokens.append(next_token)
            
            # 更新context
            context = self.update_context(context, next_token)
            
        return tokens
```

#### 3.2 实时控制

**对应论文**: Section IV.E (Robot System Details)

```python
def real_time_control_loop():
    """实时控制循环
    
    对应论文:
    "The π0.5 model directly commands target poses for the arms, 
     gripper, and torso lift, and the target base velocities at 50 Hz"
    """
    
    agent = Pi05Agent("pi05_final.ckpt")
    robot = MobileManipulator()
    
    # 控制频率
    dt = 1.0 / 50.0  # 50 Hz
    
    # 任务指令
    task = "clean the kitchen"
    
    # 初始高层推理
    observation = robot.get_observation()
    current_subtask = agent.high_level_inference(observation, task)
    
    # 预测第一个action chunk
    action_chunk = agent.low_level_inference(observation, current_subtask)
    action_idx = 0
    
    last_time = time.time()
    last_high_level_time = time.time()
    
    while True:
        current_time = time.time()
        
        # 1. 维持50Hz控制频率
        if current_time - last_time < dt:
            time.sleep(dt - (current_time - last_time))
            continue
        last_time = current_time
        
        # 2. 执行当前动作
        # 论文: "These targets are tracked with simple PD controllers"
        robot.execute_action(action_chunk[action_idx])
        action_idx += 1
        
        # 3. 获取新观察
        observation = robot.get_observation()
        
        # 4. 更新高层推理（~0.5Hz）
        if current_time - last_high_level_time >= 2.0:
            current_subtask = agent.high_level_inference(
                observation, task
            )
            last_high_level_time = current_time
            action_idx = 0  # 重置action index
            
        # 5. 如果action chunk用完，生成新的
        if action_idx >= len(action_chunk):
            action_chunk = agent.low_level_inference(
                observation, current_subtask
            )
            action_idx = 0
```

---

### 4. 数据处理 (`openpi/data/`)

#### 4.1 数据集加载器

**对应论文**: Section IV.C & IV.D

```python
class Pi05Dataset:
    """π0.5训练数据集
    
    支持多种数据源的混合
    """
    
    def __init__(
        self,
        data_sources: List[str],
        split: str = "train",
        augment: bool = True,
    ):
        """
        Args:
            data_sources: 数据源列表，如 ["MM", "ME", "CE", "HL", "WD"]
            split: "train" 或 "val"
            augment: 是否使用数据增强
        """
        
        self.datasets = {}
        self.augment = augment
        
        # 加载各个数据源
        for source in data_sources:
            if source == "MM":
                self.datasets["MM"] = self.load_mobile_manipulator()
            elif source == "ME":
                self.datasets["ME"] = self.load_multi_environment()
            elif source == "CE":
                self.datasets["CE"] = self.load_cross_embodiment()
            elif source == "HL":
                self.datasets["HL"] = self.load_high_level()
            elif source == "WD":
                self.datasets["WD"] = self.load_web_data()
            elif source == "VI":
                self.datasets["VI"] = self.load_verbal_instructions()
                
    def __getitem__(self, idx):
        """采样一个batch
        
        根据数据源权重随机采样
        """
        
        # 1. 选择数据源
        source = self.sample_source()
        
        # 2. 从该数据源采样
        data = self.datasets[source].sample()
        
        # 3. 数据增强
        if self.augment:
            data = self.apply_augmentation(data)
            
        return data
        
    def load_mobile_manipulator(self):
        """加载MM数据
        
        对应论文 Section IV.C:
        "about 400 hours of data of mobile manipulators performing 
         household tasks in about 100 different home environments"
        """
        
        # 数据格式:
        # - images: [T, 4, H, W, 3]  (4 cameras)
        # - actions: [T, 19]         (action dim)
        # - language: str            (task description)
        # - subtasks: List[str]      (if HL annotation)
        
        episodes = load_from_disk("data/mobile_manipulator/")
        return EpisodeDataset(episodes)
        
    def load_web_data(self):
        """加载WD数据
        
        对应论文 Section IV.C:
        "image captioning (CapsFusion, COCO), question answering 
         (Cambrian-7M, PixMo, VQAv2), and object localization"
        """
        
        # 数据格式:
        # - image: [H, W, 3]
        # - question: str
        # - answer: str
        # - bboxes: Optional[List[Tuple]]
        
        datasets = []
        datasets.append(load_capsfusion())
        datasets.append(load_coco())
        datasets.append(load_vqa())
        datasets.append(load_object_localization())
        
        return ConcatDataset(datasets)
        
    def apply_augmentation(self, data):
        """数据增强
        
        对应论文 Appendix E:
        "random crop, resizing, rotation, and color jittering"
        """
        
        if 'images' in data:
            # 图像增强
            # 论文中的精确参数
            transforms = [
                RandomCrop(0.95),                    # 95% crop
                Resize(original_size),
                Rotate(low=-5, high=5),              # ±5度
                ColorJitter(
                    brightness=0.3,
                    contrast=0.4,
                    saturation=0.5
                ),
            ]
            
            data['images'] = apply_transforms(data['images'], transforms)
            
        return data
```

#### 4.2 FAST Tokenizer

**对应论文**: FAST paper [64]

```python
class FASTTokenizer:
    """FAST: Efficient Action Tokenization
    
    使用DCT进行动作压缩
    """
    
    def __init__(
        self,
        action_dim: int,
        codebook_size: int = 1024,
        num_codes: int = 8,
    ):
        """
        Args:
            action_dim: 动作维度
            codebook_size: codebook大小
            num_codes: 每个动作chunk用几个token
        """
        
        self.action_dim = action_dim
        self.codebook_size = codebook_size
        self.num_codes = num_codes
        
        # DCT矩阵
        self.dct_matrix = self.build_dct_matrix()
        
        # Learned codebook
        self.codebook = nn.Embedding(codebook_size, action_dim)
        
    def encode(self, actions):
        """编码动作为离散token
        
        流程: 动作 → 归一化 → DCT → 量化 → tokens
        """
        
        # 1. 归一化到[-1, 1]
        # 论文: "1% and 99% quantile of each action dimension"
        normalized = self.normalize_actions(actions)
        
        # 2. DCT变换
        dct_coeffs = self.apply_dct(normalized)  # [B, H, D]
        
        # 3. 保留最重要的系数
        # 论文: "insignificant coefficients are removed"
        important_coeffs = dct_coeffs[:, :self.num_codes]  # [B, num_codes, D]
        
        # 4. 量化到codebook
        tokens = self.quantize_to_codebook(important_coeffs)  # [B, num_codes]
        
        return tokens
        
    def decode(self, tokens):
        """解码token为动作"""
        
        # 1. 从codebook查找
        dct_coeffs = self.codebook(tokens)  # [B, num_codes, D]
        
        # 2. 补零到完整长度
        padded_coeffs = jnp.pad(
            dct_coeffs,
            ((0,0), (0, self.action_horizon - self.num_codes), (0,0))
        )
        
        # 3. 逆DCT
        actions = self.apply_idct(padded_coeffs)
        
        # 4. 反归一化
        actions = self.denormalize_actions(actions)
        
        return actions
        
    def apply_dct(self, actions):
        """应用DCT变换
        
        DCT-II: X_k = Σ_n x_n · cos(π·k·(n+0.5)/N)
        """
        # 使用预计算的DCT矩阵
        return jnp.matmul(actions, self.dct_matrix.T)
```

---

### 5. 关键函数与论文对应

| 函数/类 | 代码位置 | 论文章节 | 关键内容 |
|---------|---------|----------|----------|
| `Pi05Policy` | `policies/pi05.py` | IV.A | 主模型架构 |
| `ActionExpert` | `models/action_expert.py` | IV.A, Appendix E | Action Expert实现 |
| `AdaptiveRMSNorm` | `models/attention.py` | Appendix E | 时间步注入 |
| `flow_matching_train` | `policies/pi05.py` | IV.B | Flow matching训练 |
| `flow_matching_inference` | `policies/pi05.py` | IV.B | 迭代去噪 |
| `create_attention_mask` | `policies/pi05.py` | Appendix E, Fig 18 | 注意力mask |
| `train_pi05` | `training/train.py` | IV.C, IV.D | 两阶段训练 |
| `compute_pretrain_loss` | `training/losses.py` | Equation (1), α=0 | Pre-training损失 |
| `compute_posttrain_loss` | `training/losses.py` | Equation (1), α=10 | Post-training损失 |
| `high_level_inference` | `policies/pi05.py` | V | 高层推理 |
| `low_level_inference` | `policies/pi05.py` | V | 低层推理 |
| `FASTTokenizer` | `data/tokenizers/fast.py` | IV.B, [64] | FAST编码 |
| `sample_flow_timestep` | `training/train.py` | Appendix E | Beta采样 |
| `Pi05Dataset` | `data/dataset.py` | IV.C, IV.D | 数据加载 |

---

## 代码使用示例

### 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi

# 2. 安装依赖
pip install -e .

# 3. 下载预训练模型
python scripts/download_models.py --model pi05_base

# 4. 运行推理
python scripts/inference.py \
    --model pi05_base \
    --task "clean the kitchen" \
    --robot sim
```

### 微调到自定义任务

```bash
# 1. 准备数据集（LeRobot格式）
python scripts/convert_data_to_lerobot.py \
    --input_dir /path/to/your/data \
    --output_dir ./datasets/my_task

# 2. 计算归一化统计
python scripts/compute_norm_stats.py \
    --dataset my_task

# 3. 微调
python scripts/train.py \
    --config configs/pi05_finetune.yaml \
    --dataset my_task \
    --base_model pi05_base \
    --num_steps 10000
```

### 评估

```bash
# 在仿真环境中评估
python scripts/eval.py \
    --model pi05_my_task \
    --env simpler \
    --task pick_horizontal_coke_can \
    --num_episodes 10
```

---

## 调试技巧

### 1. 可视化注意力

```python
from openpi.visualization import plot_attention_map

# 在forward中保存attention weights
attn_weights = model.save_attention_weights()

# 可视化
plot_attention_map(
    attn_weights,
    tokens=['img1', 'img2', 'text1', 'action1'],
    save_path='attention.png'
)
```

### 2. 检查action预测

```python
# 比较FAST和Flow Matching的预测
fast_actions = model.fast_tokenizer.decode(fast_tokens)
flow_actions = model.flow_matching_inference(vlm_output)

# 可视化差异
plot_action_comparison(fast_actions, flow_actions)
```

### 3. 监控训练

```python
import wandb

# 初始化wandb
wandb.init(project="pi05-training")

# 记录metrics
wandb.log({
    "train/text_loss": text_loss,
    "train/flow_loss": flow_loss,
    "train/total_loss": total_loss,
    "learning_rate": current_lr,
})
```

---

## 常见问题

### Q1: 为什么我的GPU内存不够？

**A**: π0.5需要大量GPU内存。解决方案：
1. 使用gradient checkpointing
2. 减小batch size
3. 使用FSDP（Fully Sharded Data Parallel）

```python
# 启用gradient checkpointing
model = Pi05Policy(use_gradient_checkpointing=True)

# 使用FSDP
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

sharding = PositionalSharding(mesh_utils.create_device_mesh((4,)))
model = jax.device_put(model, sharding)
```

### Q2: Pre-training太慢怎么办？

**A**: 
1. 确保使用多GPU
2. 使用bf16混合精度
3. 增大batch size（配合gradient accumulation）

```python
# 混合精度训练
from jax import config
config.update("jax_enable_x64", False)  # 使用bf16

# Gradient accumulation
for micro_step in range(grad_accum_steps):
    loss += compute_loss(model, micro_batch) / grad_accum_steps
```

### Q3: 如何判断模型收敛？

**A**: 监控以下指标：
1. Text loss下降到1-2
2. Flow matching loss下降到0.01-0.05
3. 在验证集上的success rate稳定

---

## 扩展阅读

### 相关论文实现

1. **PaliGemma**: https://github.com/google-research/paligemma
2. **FAST Tokenizer**: 参考π0.5的实现
3. **Flow Matching**: https://github.com/atong01/conditional-flow-matching

### 推荐资源

1. **论文精读**: https://arxiv.org/abs/2504.16054
2. **官方博客**: https://www.pi.website/blog/pi05  
3. **HuggingFace模型**: https://huggingface.co/lerobot/pi05_base
4. **LeRobot文档**: https://github.com/huggingface/lerobot

---

## 总结

π0.5的代码实现核心要点：

1. **模块化设计**: VLM + Action Expert分离
2. **混合训练**: 离散token（pre-train）+ 连续flow（post-train）
3. **层次推理**: 高层子任务 + 低层动作
4. **高效推理**: 10步去噪 + KV cache
5. **数据多样性**: 5+种数据源融合

**最重要的三个文件**:
1. `policies/pi05.py` - 模型架构
2. `training/train.py` - 训练流程  
3. `scripts/inference.py` - 推理接口

通过理解这三个文件，你就能掌握π0.5的核心实现！
