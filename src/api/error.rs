// API 错误处理模块
// 提供统一的错误响应转换、错误中间件、错误上下文

use crate::api::dto::ErrorResponse;
use crate::types::Error;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use log::{error, warn};
use serde::Serialize;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// 统一的 API 错误响应
#[derive(Debug, Serialize)]
pub struct ApiErrorResponse {
    pub error: String,
    pub code: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
    pub status: u16,
    pub timestamp: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

impl ApiErrorResponse {
    pub fn new(error: impl Into<String>, code: impl Into<String>, status: u16) -> Self {
        Self {
            error: error.into(),
            code: code.into(),
            details: None,
            status,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            request_id: None,
        }
    }

    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }

    pub fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    pub fn from_error(e: &Error) -> Self {
        let status = e.status_code();
        let code = e.error_code();
        let message = e.to_string();
        Self::new(message, code, status)
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let code = self.error_code().to_string();
        let message = self.to_string();

        // 根据严重程度记录日志
        if status >= 500 {
            error!("API error [{}]: {} - {}", status, code, message);
        } else if status >= 400 {
            warn!("API error [{}]: {} - {}", status, code, message);
        }

        let status_code = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let response = ApiErrorResponse::new(message, code, status);

        (status_code, Json(response)).into_response()
    }
}

impl IntoResponse for ApiErrorResponse {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.status)
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (status, Json(self)).into_response()
    }
}

/// 从字符串创建一个内部错误
pub fn internal_error(msg: impl Into<String>) -> Error {
    Error::Internal(msg.into())
}

/// 从字符串创建一个未找到错误
pub fn not_found(msg: impl Into<String>) -> Error {
    Error::NotFound(msg.into())
}

/// 从字符串创建一个错误请求错误
pub fn bad_request(msg: impl Into<String>) -> Error {
    Error::BadRequest(msg.into())
}

/// 从字符串创建一个服务不可用错误
pub fn service_unavailable(msg: impl Into<String>) -> Error {
    Error::ServiceUnavailable(msg.into())
}

/// 错误上下文 - 用于追踪错误链
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub operation: String,
    pub component: String,
    pub extra: HashMap<String, String>,
}

impl ErrorContext {
    pub fn new(operation: impl Into<String>, component: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            component: component.into(),
            extra: HashMap::new(),
        }
    }

    pub fn with_extra(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }

    pub fn wrap_error(&self, e: &Error) -> Error {
        let mut msg = format!("[{}] {}: {}", self.component, self.operation, e);
        if !self.extra.is_empty() {
            let extras: Vec<String> = self
                .extra
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            msg.push_str(&format!(" ({})", extras.join(", ")));
        }
        Error::Internal(msg)
    }
}

/// 从 Result 中提取错误响应
pub fn handle_result<T: IntoResponse>(
    result: Result<T, Error>,
) -> Response {
    match result {
        Ok(r) => r.into_response(),
        Err(e) => e.into_response(),
    }
}

/// 将 ErrorResponse 转换为带状态码的 Response
pub fn error_response(
    status: StatusCode,
    error: impl Into<String>,
    code: impl Into<String>,
) -> Response {
    let response = ErrorResponse::new(error, code);
    (status, Json(response)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_status_codes() {
        assert_eq!(Error::NotFound("test".to_string()).status_code(), 404);
        assert_eq!(Error::Unauthorized("test".to_string()).status_code(), 401);
        assert_eq!(Error::Forbidden("test".to_string()).status_code(), 403);
        assert_eq!(Error::Conflict("test".to_string()).status_code(), 409);
        assert_eq!(Error::Timeout("test".to_string()).status_code(), 408);
        assert_eq!(Error::BadRequest("test".to_string()).status_code(), 400);
        assert_eq!(Error::ServiceUnavailable("test".to_string()).status_code(), 503);
        assert_eq!(Error::Internal("test".to_string()).status_code(), 500);
        assert_eq!(Error::ValidationFailed("test".to_string()).status_code(), 422);
        assert_eq!(Error::TypeError("test".to_string()).status_code(), 400);
    }

    #[test]
    fn test_error_codes() {
        assert_eq!(Error::NotFound("x".to_string()).error_code(), "NOT_FOUND");
        assert_eq!(Error::Internal("x".to_string()).error_code(), "INTERNAL_ERROR");
        assert_eq!(Error::Timeout("x".to_string()).error_code(), "TIMEOUT");
        assert_eq!(Error::BackendError("x".to_string()).error_code(), "BACKEND_ERROR");
    }

    #[test]
    fn test_is_retryable() {
        assert!(Error::Timeout("x".to_string()).is_retryable());
        assert!(Error::ServiceUnavailable("x".to_string()).is_retryable());
        assert!(Error::BackendError("x".to_string()).is_retryable());
        assert!(!Error::NotFound("x".to_string()).is_retryable());
        assert!(!Error::BadRequest("x".to_string()).is_retryable());
    }

    #[test]
    fn test_api_error_response_creation() {
        let resp = ApiErrorResponse::new("Test error", "TEST_CODE", 400);
        assert_eq!(resp.error, "Test error");
        assert_eq!(resp.code, "TEST_CODE");
        assert_eq!(resp.status, 400);
        assert!(resp.details.is_none());
        assert!(resp.request_id.is_none());
    }

    #[test]
    fn test_api_error_response_with_options() {
        let resp = ApiErrorResponse::new("Test error", "TEST_CODE", 400)
            .with_details("Some details")
            .with_request_id("req-123");
        assert_eq!(resp.details, Some("Some details".to_string()));
        assert_eq!(resp.request_id, Some("req-123".to_string()));
    }

    #[test]
    fn test_api_error_response_from_error() {
        let error = Error::NotFound("Resource missing".to_string());
        let resp = ApiErrorResponse::from_error(&error);
        assert_eq!(resp.status, 404);
        assert_eq!(resp.code, "NOT_FOUND");
        assert!(resp.error.contains("Resource missing"));
    }

    #[test]
    fn test_error_context() {
        let ctx = ErrorContext::new("load_model", "CheckpointLoader")
            .with_extra("model", "sd-v1-5")
            .with_extra("step", "1");
        let err = Error::IoError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));
        let wrapped = ctx.wrap_error(&err);
        let msg = wrapped.to_string();
        assert!(msg.contains("CheckpointLoader"));
        assert!(msg.contains("load_model"));
        assert!(msg.contains("model=sd-v1-5"));
        assert!(msg.contains("step=1"));
    }

    #[test]
    fn test_error_helper_functions() {
        let e = internal_error("oops");
        assert!(matches!(e, Error::Internal(_)));

        let e = not_found("missing");
        assert!(matches!(e, Error::NotFound(_)));

        let e = bad_request("invalid");
        assert!(matches!(e, Error::BadRequest(_)));

        let e = service_unavailable("down");
        assert!(matches!(e, Error::ServiceUnavailable(_)));
    }
}
