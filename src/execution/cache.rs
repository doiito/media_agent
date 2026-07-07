// 层级缓存系统

use crate::types::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// 层级缓存 (节点级缓存)
pub struct HierarchicalCache {
    /// 节点输出缓存
    node_cache: Arc<RwLock<HashMap<NodeFingerprint, HashMap<String, Value>>>>,
    /// 模型状态缓存
    model_cache: Arc<RwLock<HashMap<String, ModelState>>>,
    /// CLIP状态缓存
    clip_cache: Arc<RwLock<HashMap<String, ClipState>>>,
    /// VAE状态缓存
    vae_cache: Arc<RwLock<HashMap<String, VaeState>>>,
}

#[derive(Debug, Clone)]
struct ModelState {
    model_id: String,
    loaded: bool,
}

#[derive(Debug, Clone)]
struct ClipState {
    clip_id: String,
    loaded: bool,
}

#[derive(Debug, Clone)]
struct VaeState {
    vae_id: String,
    loaded: bool,
}

impl HierarchicalCache {
    pub fn new() -> Self {
        Self {
            node_cache: Arc::new(RwLock::new(HashMap::new())),
            model_cache: Arc::new(RwLock::new(HashMap::new())),
            clip_cache: Arc::new(RwLock::new(HashMap::new())),
            vae_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 获取节点缓存
    pub fn get(&self, fingerprint: &NodeFingerprint) -> Option<HashMap<String, Value>> {
        let cache = self.node_cache.read().unwrap();
        cache.get(fingerprint).cloned()
    }

    /// 存入节点缓存
    pub fn put(&self, fingerprint: NodeFingerprint, output: HashMap<String, Value>) {
        let mut cache = self.node_cache.write().unwrap();
        cache.insert(fingerprint, output);
    }

    /// 检查缓存是否存在
    pub fn contains(&self, fingerprint: &NodeFingerprint) -> bool {
        let cache = self.node_cache.read().unwrap();
        cache.contains_key(fingerprint)
    }

    /// 清空节点缓存
    pub fn clear(&mut self) {
        let mut cache = self.node_cache.write().unwrap();
        cache.clear();
    }

    /// 清空模型缓存
    pub fn clear_models(&mut self) {
        let mut cache = self.model_cache.write().unwrap();
        cache.clear();
    }

    /// 清空CLIP缓存
    pub fn clear_clips(&mut self) {
        let mut cache = self.clip_cache.write().unwrap();
        cache.clear();
    }

    /// 清空VAE缓存
    pub fn clear_vaes(&mut self) {
        let mut cache = self.vae_cache.write().unwrap();
        cache.clear();
    }

    /// 获取缓存大小
    pub fn size(&self) -> usize {
        let cache = self.node_cache.read().unwrap();
        cache.len()
    }

    /// 获取模型缓存状态
    pub fn get_model_state(&self, model_id: &str) -> Option<bool> {
        let cache = self.model_cache.read().unwrap();
        cache.get(model_id).map(|s| s.loaded)
    }

    /// 设置模型缓存状态
    pub fn set_model_state(&self, model_id: String, loaded: bool) {
        let mut cache = self.model_cache.write().unwrap();
        cache.insert(model_id.clone(), ModelState { model_id, loaded });
    }

    /// 获取CLIP缓存状态
    pub fn get_clip_state(&self, clip_id: &str) -> Option<bool> {
        let cache = self.clip_cache.read().unwrap();
        cache.get(clip_id).map(|s| s.loaded)
    }

    /// 设置CLIP缓存状态
    pub fn set_clip_state(&self, clip_id: String, loaded: bool) {
        let mut cache = self.clip_cache.write().unwrap();
        cache.insert(clip_id.clone(), ClipState { clip_id, loaded });
    }

    /// 获取VAE缓存状态
    pub fn get_vae_state(&self, vae_id: &str) -> Option<bool> {
        let cache = self.vae_cache.read().unwrap();
        cache.get(vae_id).map(|s| s.loaded)
    }

    /// 设置VAE缓存状态
    pub fn set_vae_state(&self, vae_id: String, loaded: bool) {
        let mut cache = self.vae_cache.write().unwrap();
        cache.insert(vae_id.clone(), VaeState { vae_id, loaded });
    }
}

impl Clone for HierarchicalCache {
    fn clone(&self) -> Self {
        Self {
            node_cache: self.node_cache.clone(),
            model_cache: self.model_cache.clone(),
            clip_cache: self.clip_cache.clone(),
            vae_cache: self.vae_cache.clone(),
        }
    }
}