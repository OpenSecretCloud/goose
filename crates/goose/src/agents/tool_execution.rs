use async_stream::try_stream;
use futures::stream::{self, BoxStream};
use futures::{Stream, StreamExt};
use rmcp::model::CallToolResult;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use std::path::PathBuf;

use crate::config::permission::PermissionLevel;
use crate::conversation::message::{Message, ToolResponseProvenance};
#[cfg(test)]
pub(crate) use crate::conversation::message::{
    CANCELLED_RESPONSE, CHAT_MODE_TOOL_SKIPPED_RESPONSE, DECLINED_RESPONSE,
};
use crate::mcp_utils::ToolResult;
use crate::permission::permission_confirmation::PrincipalType;
use crate::permission::{Permission, PermissionConfirmation};
use rmcp::model::ServerNotification;

#[derive(Clone)]
pub(crate) struct ToolCallNotificationEmitter {
    sender: mpsc::Sender<ServerNotification>,
}

impl ToolCallNotificationEmitter {
    pub(crate) fn new(sender: mpsc::Sender<ServerNotification>) -> Self {
        Self { sender }
    }

    pub(crate) fn emit_best_effort(&self, notification: ServerNotification) {
        // Do not let a slow notification consumer delay tool execution.
        let _ = self.sender.try_send(notification);
    }
}

/// Context passed through the tool call dispatch chain.
#[derive(Clone)]
pub struct ToolCallContext {
    pub session_id: String,
    pub working_dir: Option<PathBuf>,
    pub tool_call_request_id: Option<String>,
    notification_emitter: Option<ToolCallNotificationEmitter>,
}

impl ToolCallContext {
    pub fn new(
        session_id: String,
        working_dir: Option<PathBuf>,
        tool_call_request_id: Option<String>,
    ) -> Self {
        Self {
            session_id,
            working_dir,
            tool_call_request_id,
            notification_emitter: None,
        }
    }

    pub fn working_dir_str(&self) -> Option<&str> {
        self.working_dir.as_ref().and_then(|p| p.to_str())
    }

    pub(crate) fn with_notification_emitter(
        mut self,
        notification_emitter: ToolCallNotificationEmitter,
    ) -> Self {
        self.notification_emitter = Some(notification_emitter);
        self
    }

    pub(crate) fn notification_emitter(&self) -> Option<&ToolCallNotificationEmitter> {
        self.notification_emitter.as_ref()
    }
}

// ToolCallResult combines the result of a tool call with an optional notification stream that
// can be used to receive notifications from the tool.
pub struct ToolCallResult {
    pub result: Box<dyn Future<Output = ToolResult<rmcp::model::CallToolResult>> + Send + Unpin>,
    pub notification_stream: Option<Box<dyn Stream<Item = ServerNotification> + Send + Unpin>>,
    pub action_required_stream: Option<Box<dyn Stream<Item = Message> + Send + Unpin>>,
}

impl From<ToolResult<rmcp::model::CallToolResult>> for ToolCallResult {
    fn from(result: ToolResult<rmcp::model::CallToolResult>) -> Self {
        Self {
            result: Box::new(futures::future::ready(result)),
            notification_stream: None,
            action_required_stream: None,
        }
    }
}

use crate::agents::Agent;
use crate::conversation::message::ToolRequest;
use crate::session::Session;
use crate::tool_inspection::get_security_finding_id_from_results;

pub(super) enum ToolStreamItem<T> {
    ActionRequired(Message),
    Message(ServerNotification),
    Result(T),
}

pub(super) type ToolStream =
    Pin<Box<dyn Stream<Item = ToolStreamItem<ToolResult<CallToolResult>>> + Send>>;

pub(super) fn tool_stream<S, A, F>(rx: S, action_required_rx: A, done: F) -> ToolStream
where
    S: Stream<Item = ServerNotification> + Send + Unpin + 'static,
    A: Stream<Item = Message> + Send + Unpin + 'static,
    F: Future<Output = ToolResult<CallToolResult>> + Send + 'static,
{
    Box::pin(async_stream::stream! {
        tokio::pin!(done);
        let mut rx = rx;
        let mut action_required_rx = action_required_rx;

        loop {
            tokio::select! {
                Some(msg) = action_required_rx.next() => {
                    yield ToolStreamItem::ActionRequired(msg);
                }
                Some(msg) = rx.next() => {
                    yield ToolStreamItem::Message(msg);
                }
                r = &mut done => {
                    yield ToolStreamItem::Result(r);
                    break;
                }
            }
        }
    })
}

struct ConfirmationOutcome {
    confirmation: PermissionConfirmation,
    cancelled_by_token: bool,
}

async fn await_confirmation_or_cancel(
    request_id: &str,
    confirmation_rx: oneshot::Receiver<PermissionConfirmation>,
    cancellation_token: Option<&CancellationToken>,
) -> anyhow::Result<ConfirmationOutcome> {
    let receive = async {
        confirmation_rx
            .await
            .map_err(|_| anyhow::anyhow!("Confirmation channel closed for request {}", request_id))
    };
    if let Some(cancellation_token) = cancellation_token {
        tokio::select! {
            biased;
            _ = cancellation_token.cancelled() => Ok(ConfirmationOutcome {
                confirmation: PermissionConfirmation {
                    principal_type: PrincipalType::Tool,
                    permission: Permission::Cancel,
                },
                cancelled_by_token: true,
            }),
            confirmation = receive => confirmation.map(|confirmation| ConfirmationOutcome {
                confirmation,
                cancelled_by_token: false,
            }),
        }
    } else {
        receive.await.map(|confirmation| ConfirmationOutcome {
            confirmation,
            cancelled_by_token: false,
        })
    }
}

impl Agent {
    pub(super) fn handle_approval_tool_requests<'a>(
        &'a self,
        tool_requests: &'a [ToolRequest],
        tool_futures: &'a mut Vec<(String, ToolStream)>,
        request_to_response_map: &'a mut HashMap<String, Message>,
        cancellation_token: Option<CancellationToken>,
        session: &'a Session,
        inspection_results: &'a [crate::tool_inspection::InspectionResult],
    ) -> BoxStream<'a, anyhow::Result<Message>> {
        try_stream! {
        for request in tool_requests.iter() {
            if let Ok(tool_call) = request.tool_call.clone() {
                if cancellation_token
                    .as_ref()
                    .is_some_and(CancellationToken::is_cancelled)
                {
                    if let Some(response) = request_to_response_map.get_mut(&request.id) {
                        response.add_goose_control_tool_response_with_metadata(
                            request.id.clone(),
                            ToolResponseProvenance::GooseCancelledBeforeExecution,
                            request.metadata.as_ref(),
                        );
                    }
                    continue;
                }

                let security_message = inspection_results.iter()
                    .find(|result| result.tool_request_id == request.id)
                    .and_then(|result| {
                        if let crate::tool_inspection::InspectionAction::RequireApproval(Some(message)) = &result.action {
                            Some(message.clone())
                        } else {
                            None
                        }
                    });

                let confirmation_rx = self.tool_confirmation_router.register(request.id.clone()).await;

                let action_required_msg = Message::assistant()
                    .with_action_required(
                        request.id.clone(),
                        tool_call.name.to_string().clone(),
                        tool_call.arguments.clone().unwrap_or_default(),
                        security_message,
                    )
                    .user_only();
                yield action_required_msg;

                let confirmation_outcome = await_confirmation_or_cancel(
                    &request.id,
                    confirmation_rx,
                    cancellation_token.as_ref(),
                )
                .await?;
                let confirmation = confirmation_outcome.confirmation;

                if !confirmation_outcome.cancelled_by_token {
                    if let Some(finding_id) =
                        get_security_finding_id_from_results(&request.id, inspection_results)
                    {
                        let action = match confirmation.permission {
                            Permission::AllowOnce | Permission::AlwaysAllow => "ALLOW",
                            _ => "BLOCK",
                        };
                        tracing::info!(
                            monotonic_counter.goose.prompt_injection_user_decisions = 1,
                            security.event_type = "user_decision",
                            security.action = action,
                            security.finding_id = %finding_id,
                            tool.request_id = %request.id,
                            user.decision = ?confirmation.permission,
                            "security finding: user decision"
                        );
                    }
                }

                let cancelled = cancellation_token
                    .as_ref()
                    .is_some_and(CancellationToken::is_cancelled);
                if !cancelled
                    && (confirmation.permission == Permission::AllowOnce
                        || confirmation.permission == Permission::AlwaysAllow)
                {
                    let (req_id, tool_result) = self.dispatch_tool_call(tool_call.clone(), request.id.clone(), cancellation_token.clone(), session).await;

                    tool_futures.push((req_id, match tool_result {
                        Ok(result) => tool_stream(
                            result.notification_stream.unwrap_or_else(|| Box::new(stream::empty())),
                            result.action_required_stream.unwrap_or_else(|| Box::new(stream::empty())),
                            result.result,
                        ),
                        Err(e) => tool_stream(
                            Box::new(stream::empty()),
                            Box::new(stream::empty()),
                            futures::future::ready(Err(e)),
                        ),
                    }));

                    if confirmation.permission == Permission::AlwaysAllow {
                        self.tool_inspection_manager
                            .update_permission_manager(&tool_call.name, PermissionLevel::AlwaysAllow)
                            .await;
                    }
                } else {
                    if let Some(response) = request_to_response_map.get_mut(&request.id) {
                        let provenance = if cancelled
                            || confirmation.permission == Permission::Cancel
                        {
                            ToolResponseProvenance::GooseCancelledBeforeExecution
                        } else {
                            ToolResponseProvenance::GooseDeniedBeforeExecution
                        };
                        response.add_goose_control_tool_response_with_metadata(
                            request.id.clone(),
                            provenance,
                            request.metadata.as_ref(),
                        );
                    }

                    if confirmation.permission == Permission::AlwaysDeny {
                        self.tool_inspection_manager
                            .update_permission_manager(&tool_call.name, PermissionLevel::NeverAllow)
                            .await;
                    }
                }
            }
        }
    }.boxed()
    }

    pub(crate) fn handle_frontend_tool_request<'a>(
        &'a self,
        tool_request: &'a ToolRequest,
        message_tool_response: &'a mut Message,
    ) -> BoxStream<'a, anyhow::Result<Message>> {
        try_stream! {
                if let Ok(tool_call) = tool_request.tool_call.clone() {
                    if self.is_frontend_tool(&tool_call.name).await {
                        yield Message::assistant().with_frontend_tool_request(
                            tool_request.id.clone(),
                            Ok(tool_call.clone())
                        );

                        if let Some((id, result)) = self.tool_result_rx.lock().await.recv().await {
                            message_tool_response.add_tool_response_with_metadata(
                                id,
                                result,
                                tool_request.metadata.as_ref(),
                            );
                        }
                    }
            }
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allow_once() -> PermissionConfirmation {
        PermissionConfirmation {
            principal_type: PrincipalType::Tool,
            permission: Permission::AllowOnce,
        }
    }

    fn deny_once() -> PermissionConfirmation {
        PermissionConfirmation {
            principal_type: PrincipalType::Tool,
            permission: Permission::DenyOnce,
        }
    }

    fn cancel_once() -> PermissionConfirmation {
        PermissionConfirmation {
            principal_type: PrincipalType::Tool,
            permission: Permission::Cancel,
        }
    }

    #[tokio::test]
    async fn denied_approval_records_a_canonical_goose_control_response() {
        let agent = Agent::new();
        let request = ToolRequest {
            id: "request".to_string(),
            tool_call: Ok(rmcp::model::CallToolRequestParams::new(
                "missing_extension__tool".to_string(),
            )),
            metadata: None,
            tool_meta: None,
        };
        let mut tool_futures = Vec::new();
        let mut responses =
            HashMap::from([(request.id.clone(), Message::user().with_generated_id())]);
        let session = Session {
            id: "session".to_string(),
            ..Session::default()
        };
        let inspection_results = Vec::new();
        let mut approval_stream = agent.handle_approval_tool_requests(
            std::slice::from_ref(&request),
            &mut tool_futures,
            &mut responses,
            Some(CancellationToken::new()),
            &session,
            &inspection_results,
        );

        approval_stream
            .next()
            .await
            .expect("approval stream should request confirmation")
            .expect("approval request should be valid");
        assert!(
            agent
                .tool_confirmation_router
                .deliver(request.id.clone(), deny_once())
                .await
        );
        assert!(approval_stream.next().await.is_none());
        drop(approval_stream);

        assert!(tool_futures.is_empty());
        let response = responses.get(&request.id).unwrap();
        let control = response.content.iter().find_map(|content| match content {
            crate::conversation::message::MessageContent::ToolResponse(response) => Some(response),
            _ => None,
        });
        let control = control.expect("denial should create a tool response");
        assert_eq!(
            control.provenance,
            ToolResponseProvenance::GooseDeniedBeforeExecution
        );
        assert!(control.is_canonical_goose_control_response());
    }

    #[tokio::test]
    async fn explicit_cancel_records_a_canonical_goose_cancellation_response() {
        let agent = Agent::new();
        let request = ToolRequest {
            id: "request".to_string(),
            tool_call: Ok(rmcp::model::CallToolRequestParams::new(
                "missing_extension__tool".to_string(),
            )),
            metadata: None,
            tool_meta: None,
        };
        let mut tool_futures = Vec::new();
        let mut responses =
            HashMap::from([(request.id.clone(), Message::user().with_generated_id())]);
        let session = Session {
            id: "session".to_string(),
            ..Session::default()
        };
        let mut approval_stream = agent.handle_approval_tool_requests(
            std::slice::from_ref(&request),
            &mut tool_futures,
            &mut responses,
            Some(CancellationToken::new()),
            &session,
            &[],
        );

        approval_stream
            .next()
            .await
            .expect("approval stream should request confirmation")
            .expect("approval request should be valid");
        assert!(
            agent
                .tool_confirmation_router
                .deliver(request.id.clone(), cancel_once())
                .await
        );
        assert!(approval_stream.next().await.is_none());
        drop(approval_stream);

        assert!(tool_futures.is_empty());
        let control = responses[&request.id]
            .content
            .iter()
            .find_map(|content| match content {
                crate::conversation::message::MessageContent::ToolResponse(response) => {
                    Some(response)
                }
                _ => None,
            })
            .expect("cancellation should create a tool response");
        assert_eq!(
            control.provenance,
            ToolResponseProvenance::GooseCancelledBeforeExecution
        );
        assert!(control.is_canonical_goose_control_response());
    }

    #[tokio::test]
    async fn queued_allow_once_loses_to_already_cancelled_run() {
        let (sender, receiver) = oneshot::channel();
        sender.send(allow_once()).unwrap();
        let cancellation = CancellationToken::new();
        cancellation.cancel();

        let outcome = await_confirmation_or_cancel("request", receiver, Some(&cancellation))
            .await
            .unwrap();

        assert_eq!(outcome.confirmation.permission, Permission::Cancel);
        assert!(outcome.cancelled_by_token);
    }

    #[tokio::test]
    async fn uncancelled_run_receives_queued_allow_once() {
        let (sender, receiver) = oneshot::channel();
        sender.send(allow_once()).unwrap();
        let cancellation = CancellationToken::new();

        let outcome = await_confirmation_or_cancel("request", receiver, Some(&cancellation))
            .await
            .unwrap();

        assert_eq!(outcome.confirmation.permission, Permission::AllowOnce);
        assert!(!outcome.cancelled_by_token);
    }

    #[tokio::test]
    async fn cancelled_queued_approval_never_creates_a_tool_future() {
        let agent = Agent::new();
        let request = ToolRequest {
            id: "request".to_string(),
            tool_call: Ok(rmcp::model::CallToolRequestParams::new(
                "missing_extension__tool".to_string(),
            )),
            metadata: None,
            tool_meta: None,
        };
        let second_request = ToolRequest {
            id: "second-request".to_string(),
            tool_call: Ok(rmcp::model::CallToolRequestParams::new(
                "missing_extension__second_tool".to_string(),
            )),
            metadata: None,
            tool_meta: None,
        };
        let mut tool_futures = Vec::new();
        let mut responses = HashMap::from([
            (request.id.clone(), Message::user().with_generated_id()),
            (
                second_request.id.clone(),
                Message::user().with_generated_id(),
            ),
        ]);
        let cancellation = CancellationToken::new();
        let session = Session {
            id: "session".to_string(),
            ..Session::default()
        };
        let inspection_results = Vec::new();
        let requests = [request.clone(), second_request.clone()];
        let mut approval_stream = agent.handle_approval_tool_requests(
            &requests,
            &mut tool_futures,
            &mut responses,
            Some(cancellation.clone()),
            &session,
            &inspection_results,
        );

        let action_required = approval_stream
            .next()
            .await
            .expect("approval stream should request confirmation")
            .expect("approval request should be valid");
        assert!(action_required.content.iter().any(|content| matches!(
            content,
            crate::conversation::message::MessageContent::ActionRequired(_)
        )));

        assert!(
            agent
                .tool_confirmation_router
                .deliver(request.id.clone(), allow_once())
                .await
        );
        cancellation.cancel();

        assert!(approval_stream.next().await.is_none());
        drop(approval_stream);
        assert!(
            tool_futures.is_empty(),
            "a cancelled queued approval must not reach tool dispatch"
        );
        let response = serde_json::to_string(responses.get("request").unwrap()).unwrap();
        assert!(response.contains(CANCELLED_RESPONSE));
        assert!(!response.contains(DECLINED_RESPONSE));
        assert!(response.contains("goose_cancelled_before_execution"));
        let second_response =
            serde_json::to_string(responses.get("second-request").unwrap()).unwrap();
        assert!(second_response.contains(CANCELLED_RESPONSE));
        assert!(!second_response.contains(DECLINED_RESPONSE));
        assert!(second_response.contains("goose_cancelled_before_execution"));
        assert!(
            !agent
                .tool_confirmation_router
                .deliver(second_request.id, allow_once())
                .await,
            "cancellation must not register a second confirmation"
        );
    }

    #[tokio::test]
    async fn cancellation_rejects_a_late_allow_once() {
        let agent = Agent::new();
        let request = ToolRequest {
            id: "request".to_string(),
            tool_call: Ok(rmcp::model::CallToolRequestParams::new(
                "missing_extension__tool".to_string(),
            )),
            metadata: None,
            tool_meta: None,
        };
        let mut tool_futures = Vec::new();
        let mut responses =
            HashMap::from([(request.id.clone(), Message::user().with_generated_id())]);
        let cancellation = CancellationToken::new();
        let session = Session {
            id: "session".to_string(),
            ..Session::default()
        };
        let inspection_results = Vec::new();
        let mut approval_stream = agent.handle_approval_tool_requests(
            std::slice::from_ref(&request),
            &mut tool_futures,
            &mut responses,
            Some(cancellation.clone()),
            &session,
            &inspection_results,
        );

        approval_stream
            .next()
            .await
            .expect("approval stream should request confirmation")
            .expect("approval request should be valid");
        cancellation.cancel();
        assert!(approval_stream.next().await.is_none());
        drop(approval_stream);

        assert!(tool_futures.is_empty());
        assert!(
            !agent
                .tool_confirmation_router
                .deliver(request.id.clone(), allow_once())
                .await,
            "a late approval must not revive a cancelled request"
        );
        let response = serde_json::to_string(responses.get(&request.id).unwrap()).unwrap();
        assert!(response.contains(CANCELLED_RESPONSE));
        assert!(!response.contains(DECLINED_RESPONSE));
        assert!(response.contains("goose_cancelled_before_execution"));
    }
}
