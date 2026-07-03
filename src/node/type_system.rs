// 节点类型系统
// 定义每个节点的输入输出类型约束，支持智能连线验证

use std::collections::HashMap;

/// 数据类型枚举
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DataType {
    /// 模型权重
    MODEL,
    /// CLIP 文本编码器
    CLIP,
    /// VAE 编解码器
    VAE,
    /// Conditioning 条件向量
    CONDITIONING,
    /// Latent 潜空间数据
    LATENT,
    /// 图片数据
    IMAGE,
    /// ControlNet 模型
    CONTROL_NET,
    /// LoRA 模型
    LORA,
    /// 数字
    INT,
    FLOAT,
    /// 文本
    STRING,
    /// 视频帧序列
    VIDEO_FRAMES,
    /// 任意类型（用于灵活节点）
    ANY,
}

impl DataType {
    /// 检查是否可以连接（类型兼容性）
    pub fn can_connect_to(&self, target: &DataType) -> bool {
        // ANY 可以连接任何类型
        if *self == DataType::ANY || *target == DataType::ANY {
            return true;
        }
        // 同类型可以连接
        if self == target {
            return true;
        }
        // 特殊兼容规则
        match (self, target) {
            // INT/FLOAT 可以互相连接（自动转换）
            (DataType::INT, DataType::FLOAT) => true,
            (DataType::FLOAT, DataType::INT) => true,
            // IMAGE 可以转换为 LATENT（通过 VAEEncode）
            (DataType::IMAGE, DataType::LATENT) => true,
            // LATENT 可以转换为 IMAGE（通过 VAEDecode）
            (DataType::LATENT, DataType::IMAGE) => true,
            _ => false,
        }
    }

    /// 类型名称
    pub fn name(&self) -> &str {
        match self {
            DataType::MODEL => "MODEL",
            DataType::CLIP => "CLIP",
            DataType::VAE => "VAE",
            DataType::CONDITIONING => "CONDITIONING",
            DataType::LATENT => "LATENT",
            DataType::IMAGE => "IMAGE",
            DataType::CONTROL_NET => "CONTROL_NET",
            DataType::LORA => "LORA",
            DataType::INT => "INT",
            DataType::FLOAT => "FLOAT",
            DataType::STRING => "STRING",
            DataType::VIDEO_FRAMES => "VIDEO_FRAMES",
            DataType::ANY => "*",
        }
    }
}

/// 节点端口定义
#[derive(Debug, Clone)]
pub struct PortDefinition {
    /// 端口名称
    pub name: String,
    /// 数据类型
    pub data_type: DataType,
    /// 是否必需
    pub required: bool,
    /// 默认值（可选）
    pub default: Option<String>,
    /// 描述
    pub description: String,
}

/// 节点类型定义
#[derive(Debug, Clone)]
pub struct NodeDefinition {
    /// 节点类型名称（class_type）
    pub class_type: String,
    /// 节点分类
    pub category: String,
    /// 输入端口列表
    pub inputs: Vec<PortDefinition>,
    /// 输出端口列表
    pub outputs: Vec<PortDefinition>,
    /// 节点描述
    pub description: String,
    /// 节点标签（用于智能搜索）
    pub tags: Vec<String>,
}

/// 节点类型注册表
pub struct NodeTypeRegistry {
    /// 所有节点定义
    definitions: HashMap<String, NodeDefinition>,
    /// 按分类索引
    category_index: HashMap<String, Vec<String>>,
}

impl NodeTypeRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            definitions: HashMap::new(),
            category_index: HashMap::new(),
        };
        
        // 注册所有核心节点
        registry.register_core_nodes();
        registry
    }

    /// 注册节点定义
    fn register(&mut self, def: NodeDefinition) {
        // 添加到定义表
        self.definitions.insert(def.class_type.clone(), def.clone());
        
        // 添加到分类索引
        self.category_index
            .entry(def.category.clone())
            .or_insert_with(Vec::new)
            .push(def.class_type.clone());
    }

    /// 获取节点定义
    pub fn get(&self, class_type: &str) -> Option<&NodeDefinition> {
        self.definitions.get(class_type)
    }

    /// 按分类获取节点列表
    pub fn get_by_category(&self, category: &str) -> Vec<&NodeDefinition> {
        self.category_index
            .get(category)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default()
    }

    /// 按标签搜索节点
    pub fn search_by_tag(&self, tag: &str) -> Vec<&NodeDefinition> {
        self.definitions
            .values()
            .filter(|def| def.tags.iter().any(|t| t.contains(tag)))
            .collect()
    }

    /// 获取所有节点定义
    pub fn all_definitions(&self) -> &HashMap<String, NodeDefinition> {
        &self.definitions
    }

    /// 注册核心节点
    fn register_core_nodes(&mut self) {
        // === 模型加载类 ===
        self.register(NodeDefinition {
            class_type: "CheckpointLoaderSimple".to_string(),
            category: "model".to_string(),
            inputs: vec![
                PortDefinition {
                    name: "ckpt_name".to_string(),
                    data_type: DataType::STRING,
                    required: true,
                    default: Some("v1-5-pruned-emaonly.safetensors"),
                    description: "模型文件名".to_string(),
                },
            ],
            outputs: vec![
                PortDefinition {
                    name: "MODEL".to_string(),
                    data_type: DataType::MODEL,
                    required: true,
                    default: None,
                    description: "UNET 模型".to_string(),
                },
                PortDefinition {
                    name: "CLIP".to_string(),
                    data_type: DataType::CLIP,
                    required: true,
                    default: None,
                    description: "CLIP 文本编码器".to_string(),
                },
                PortDefinition {
                    name: "VAE".to_string(),
                    data_type: DataType::VAE,
                    required: true,
                    default: None,
                    description: "VAE 编解码器".to_string(),
                },
            ],
            description: "加载 SD Checkpoint 模型".to_string(),
            tags: vec!["model", "loader", "checkpoint".to_string()],
        });

        self.register(NodeDefinition {
            class_type: "LoraLoader".to_string(),
            category: "model".to_string(),
            inputs: vec![
                PortDefinition {
                    name: "model".to_string(),
                    data_type: DataType::MODEL,
                    required: true,
                    default: None,
                    description: "输入模型".to_string(),
                },
                PortDefinition {
                    name: "clip".to_string(),
                    data_type: DataType::CLIP,
                    required: true,
                    default: None,
                    description: "输入 CLIP".to_string(),
                },
                PortDefinition {
                    name: "lora_name".to_string(),
                    data_type: DataType::STRING,
                    required: true,
                    default: None,
                    description: "LoRA 文件名".to_string(),
                },
                PortDefinition {
                    name: "strength_model".to_string(),
                    data_type: DataType::FLOAT,
                    required: false,
                    default: Some("1.0"),
                    description: "模型强度".to_string(),
                },
                PortDefinition {
                    name: "strength_clip".to_string(),
                    data_type: DataType::FLOAT,
                    required: false,
                    default: Some("1.0"),
                    description: "CLIP 强度".to_string(),
                },
            ],
            outputs: vec![
                PortDefinition {
                    name: "MODEL".to_string(),
                    data_type: DataType::MODEL,
                    required: true,
                    default: None,
                    description: "增强后的模型".to_string(),
                },
                PortDefinition {
                    name: "CLIP".to_string(),
                    data_type: DataType::CLIP,
                    required: true,
                    default: None,
                    description: "增强后的 CLIP".to_string(),
                },
            ],
            description: "加载并应用 LoRA 模型".to_string(),
            tags: vec!["lora", "loader", "style".to_string()],
        });

        // === 提示词编码类 ===
        self.register(NodeDefinition {
            class_type: "CLIPTextEncode".to_string(),
            category: "conditioning".to_string(),
            inputs: vec![
                PortDefinition {
                    name: "text".to_string(),
                    data_type: DataType::STRING,
                    required: true,
                    default: None,
                    description: "提示词文本".to_string(),
                },
                PortDefinition {
                    name: "clip".to_string(),
                    data_type: DataType::CLIP,
                    required: true,
                    default: None,
                    description: "CLIP 编码器".to_string(),
                },
            ],
            outputs: vec![
                PortDefinition {
                    name: "CONDITIONING".to_string(),
                    data_type: DataType::CONDITIONING,
                    required: true,
                    default: None,
                    description: "条件向量".to_string(),
                },
            ],
            description: "CLIP 文本编码".to_string(),
            tags: vec!["clip", "prompt", "encode".to_string()],
        });

        // === Latent 类 ===
        self.register(NodeDefinition {
            class_type: "EmptyLatentImage".to_string(),
            category: "latent".to_string(),
            inputs: vec![
                PortDefinition {
                    name: "width".to_string(),
                    data_type: DataType::INT,
                    required: false,
                    default: Some("512"),
                    description: "宽度".to_string(),
                },
                PortDefinition {
                    name: "height".to_string(),
                    data_type: DataType::INT,
                    required: false,
                    default: Some("512"),
                    description: "高度".to_string(),
                },
                PortDefinition {
                    name: "batch_size".to_string(),
                    data_type: DataType::INT,
                    required: false,
                    default: Some("1"),
                    description: "批次大小".to_string(),
                },
            ],
            outputs: vec![
                PortDefinition {
                    name: "LATENT".to_string(),
                    data_type: DataType::LATENT,
                    required: true,
                    default: None,
                    description: "空 Latent".to_string(),
                },
            ],
            description: "创建空 Latent 图像".to_string(),
            tags: vec!["latent", "empty", "create".to_string()],
        });

        // === 采样类 ===
        self.register(NodeDefinition {
            class_type: "KSampler".to_string(),
            category: "sampling".to_string(),
            inputs: vec![
                PortDefinition {
                    name: "model".to_string(),
                    data_type: DataType::MODEL,
                    required: true,
                    default: None,
                    description: "采样模型".to_string(),
                },
                PortDefinition {
                    name: "positive".to_string(),
                    data_type: DataType::CONDITIONING,
                    required: true,
                    default: None,
                    description: "正向条件".to_string(),
                },
                PortDefinition {
                    name: "negative".to_string(),
                    data_type: DataType::CONDITIONING,
                    required: true,
                    default: None,
                    description: "负向条件".to_string(),
                },
                PortDefinition {
                    name: "latent_image".to_string(),
                    data_type: DataType::LATENT,
                    required: true,
                    default: None,
                    description: "输入 Latent".to_string(),
                },
                PortDefinition {
                    name: "seed".to_string(),
                    data_type: DataType::INT,
                    required: false,
                    default: Some("0"),
                    description: "随机种子".to_string(),
                },
                PortDefinition {
                    name: "steps".to_string(),
                    data_type: DataType::INT,
                    required: false,
                    default: Some("20"),
                    description: "采样步数".to_string(),
                },
                PortDefinition {
                    name: "cfg".to_string(),
                    data_type: DataType::FLOAT,
                    required: false,
                    default: Some("7.0"),
                    description: "CFG Scale".to_string(),
                },
                PortDefinition {
                    name: "sampler_name".to_string(),
                    data_type: DataType::STRING,
                    required: false,
                    default: Some("euler"),
                    description: "采样器名称".to_string(),
                },
                PortDefinition {
                    name: "scheduler".to_string(),
                    data_type: DataType::STRING,
                    required: false,
                    default: Some("normal"),
                    description: "调度器".to_string(),
                },
                PortDefinition {
                    name: "denoise".to_string(),
                    data_type: DataType::FLOAT,
                    required: false,
                    default: Some("1.0"),
                    description: "去噪强度".to_string(),
                },
            ],
            outputs: vec![
                PortDefinition {
                    name: "LATENT".to_string(),
                    data_type: DataType::LATENT,
                    required: true,
                    default: None,
                    description: "采样结果 Latent".to_string(),
                },
            ],
            description: "标准 KSampler 采样器".to_string(),
            tags: vec!["sampler", "ksampler", "sampling".to_string()],
        });

        // === VAE 解码类 ===
        self.register(NodeDefinition {
            class_type: "VAEDecode".to_string(),
            category: "vae".to_string(),
            inputs: vec![
                PortDefinition {
                    name: "samples".to_string(),
                    data_type: DataType::LATENT,
                    required: true,
                    default: None,
                    description: "Latent 数据".to_string(),
                },
                PortDefinition {
                    name: "vae".to_string(),
                    data_type: DataType::VAE,
                    required: true,
                    default: None,
                    description: "VAE 编解码器".to_string(),
                },
            ],
            outputs: vec![
                PortDefinition {
                    name: "IMAGE".to_string(),
                    data_type: DataType::IMAGE,
                    required: true,
                    default: None,
                    description: "解码图片".to_string(),
                },
            ],
            description: "VAE Latent 解码为图片".to_string(),
            tags: vec!["vae", "decode", "image".to_string()],
        });

        self.register(NodeDefinition {
            class_type: "VAEEncode".to_string(),
            category: "vae".to_string(),
            inputs: vec![
                PortDefinition {
                    name: "pixels".to_string(),
                    data_type: DataType::IMAGE,
                    required: true,
                    default: None,
                    description: "输入图片".to_string(),
                },
                PortDefinition {
                    name: "vae".to_string(),
                    data_type: DataType::VAE,
                    required: true,
                    default: None,
                    description: "VAE 编解码器".to_string(),
                },
            ],
            outputs: vec![
                PortDefinition {
                    name: "LATENT".to_string(),
                    data_type: DataType::LATENT,
                    required: true,
                    default: None,
                    description: "编码 Latent".to_string(),
                },
            ],
            description: "VAE 图片编码为 Latent".to_string(),
            tags: vec!["vae", "encode", "latent".to_string()],
        });

        // === 图片处理类 ===
        self.register(NodeDefinition {
            class_type: "ImageScale".to_string(),
            category: "image".to_string(),
            inputs: vec![
                PortDefinition {
                    name: "image".to_string(),
                    data_type: DataType::IMAGE,
                    required: true,
                    default: None,
                    description: "输入图片".to_string(),
                },
                PortDefinition {
                    name: "width".to_string(),
                    data_type: DataType::INT,
                    required: true,
                    default: Some("1024"),
                    description: "目标宽度".to_string(),
                },
                PortDefinition {
                    name: "height".to_string(),
                    data_type: DataType::INT,
                    required: true,
                    default: Some("1024"),
                    description: "目标高度".to_string(),
                },
            ],
            outputs: vec![
                PortDefinition {
                    name: "IMAGE".to_string(),
                    data_type: DataType::IMAGE,
                    required: true,
                    default: None,
                    description: "缩放后图片".to_string(),
                },
            ],
            description: "图片缩放".to_string(),
            tags: vec!["image", "scale", "resize".to_string()],
        });

        // === 输出类 ===
        self.register(NodeDefinition {
            class_type: "SaveImage".to_string(),
            category: "output".to_string(),
            inputs: vec![
                PortDefinition {
                    name: "images".to_string(),
                    data_type: DataType::IMAGE,
                    required: true,
                    default: None,
                    description: "要保存的图片".to_string(),
                },
                PortDefinition {
                    name: "filename_prefix".to_string(),
                    data_type: DataType::STRING,
                    required: false,
                    default: Some("ComfyUI"),
                    description: "文件名前缀".to_string(),
                },
            ],
            outputs: vec![],
            description: "保存图片到输出目录".to_string(),
            tags: vec!["output", "save", "image".to_string()],
        });

        // === 视频类 ===
        self.register(NodeDefinition {
            class_type: "SVDImageToVideo".to_string(),
            category: "video".to_string(),
            inputs: vec![
                PortDefinition {
                    name: "image".to_string(),
                    data_type: DataType::IMAGE,
                    required: true,
                    default: None,
                    description: "输入图片".to_string(),
                },
                PortDefinition {
                    name: "positive".to_string(),
