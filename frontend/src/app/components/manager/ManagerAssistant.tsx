import { useEffect, useMemo, useRef, useState } from "react";
import { MessageSquare, Mic, MoreHorizontal, Pencil, Plus, Send, Trash2 } from "lucide-react";

import { AssistantResponse, getAssistantHistory, sendAssistantQuery } from "../../services/api";
import { useAuth } from "../../contexts/AuthContext";

interface AssistantMessage extends Partial<AssistantResponse> {
  id: string;
  type: "user" | "ai";
  content: string;
  mode?: string;
}

interface ChatSession {
  id: string;
  title: string;
  messages: AssistantMessage[];
  deleted?: boolean;
}

interface ModalState {
  type: "rename" | "delete";
  chatId: string;
}

interface ChatMenuState {
  chatId: string;
  x: number;
  y: number;
}

function uid(prefix: string): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
}

function formatMessageTime(iso?: string): string | null {
  if (!iso) return null;
  try {
    return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(iso));
  } catch {
    return null;
  }
}

function deriveTitle(content: string): string {
  const normalized = content.replace(/\s+/g, " ").trim();
  if (!normalized) return "New chat";
  return normalized.length > 42 ? `${normalized.slice(0, 42)}...` : normalized;
}

export function ManagerAssistant() {
  const { user, isLoading: authLoading, isAuthenticated } = useAuth();

  const [sessions, setSessions] = useState<Record<string, ChatSession>>({});
  const [selectedChatId, setSelectedChatId] = useState<string | null>(null);
  const [isDraftNewChat, setIsDraftNewChat] = useState(false);

  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const [menuState, setMenuState] = useState<ChatMenuState | null>(null);
  const [modalState, setModalState] = useState<ModalState | null>(null);
  const [renameValue, setRenameValue] = useState("");

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (modalState) {
      const original = document.body.style.overflow;
      document.body.style.overflow = "hidden";
      return () => {
        document.body.style.overflow = original;
      };
    }
    return;
  }, [modalState]);

  useEffect(() => {
    const closeMenu = () => setMenuState(null);
    window.addEventListener("click", closeMenu);
    return () => window.removeEventListener("click", closeMenu);
  }, []);

  useEffect(() => {
    if (authLoading) return;

    if (!isAuthenticated || user?.role !== "manager") {
      setHistoryLoading(false);
      setHistoryError(null);
      setSessions({});
      setSelectedChatId(null);
      return;
    }

    let cancelled = false;
    setHistoryLoading(true);
    setHistoryError(null);

    getAssistantHistory()
      .then((history) => {
        if (cancelled) return;
        if (!history?.length) return;

        // Seed one imported session only when there are no local sessions yet.
        setSessions((prev) => {
          const active = Object.values(prev).filter((s) => !s.deleted);
          if (active.length > 0) return prev;
          const importedId = uid("chat");
          const firstUser = (history as AssistantMessage[]).find((m) => m.type === "user");
          return {
            ...prev,
            [importedId]: {
              id: importedId,
              title: firstUser ? deriveTitle(firstUser.content) : "Recent chat",
              messages: history as AssistantMessage[],
            },
          };
        });
      })
      .catch((err) => {
        console.error("Failed to load chat history:", err);
        if (!cancelled) {
          setHistoryError("Could not load saved conversation. Your session may have expired - try logging in again.");
        }
      })
      .finally(() => {
        if (!cancelled) setHistoryLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [authLoading, isAuthenticated, user?.id, user?.role]);

  const chatList = useMemo(() => {
    const list = Object.values(sessions).filter((s) => !s.deleted);
    return list.sort((a, b) => a.id.localeCompare(b.id)).reverse();
  }, [sessions]);

  useEffect(() => {
    if (isDraftNewChat) return;
    if (selectedChatId && sessions[selectedChatId] && !sessions[selectedChatId].deleted) return;
    if (chatList.length > 0) setSelectedChatId(chatList[0].id);
    else setSelectedChatId(null);
  }, [chatList, selectedChatId, sessions, isDraftNewChat]);

  const selectedSession = selectedChatId ? sessions[selectedChatId] : null;
  const visibleMessages = selectedSession?.messages ?? [];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [visibleMessages, isLoading]);

  const createNewChat = () => {
    setIsDraftNewChat(true);
    setSelectedChatId(null);
    setInput("");
    setHistoryError(null);
  };

  const openChat = (chatId: string) => {
    setIsDraftNewChat(false);
    setSelectedChatId(chatId);
  };

  const openRenameModal = (chatId: string) => {
    setRenameValue(sessions[chatId]?.title || "");
    setModalState({ type: "rename", chatId });
    setMenuState(null);
  };

  const openDeleteModal = (chatId: string) => {
    setModalState({ type: "delete", chatId });
    setMenuState(null);
  };

  const confirmRename = () => {
    if (!modalState || modalState.type !== "rename") return;
    const next = renameValue.trim();
    if (!next) return;
    setSessions((prev) => ({ ...prev, [modalState.chatId]: { ...prev[modalState.chatId], title: next } }));
    setModalState(null);
  };

  const confirmDelete = () => {
    if (!modalState || modalState.type !== "delete") return;
    setSessions((prev) => ({ ...prev, [modalState.chatId]: { ...prev[modalState.chatId], deleted: true } }));
    if (selectedChatId === modalState.chatId) setSelectedChatId(null);
    setModalState(null);
    setIsDraftNewChat(false);
  };

  const appendToSession = (chatId: string, message: AssistantMessage) => {
    setSessions((prev) => {
      const existing = prev[chatId];
      if (!existing) return prev;
      return {
        ...prev,
        [chatId]: {
          ...existing,
          messages: [...existing.messages, message],
        },
      };
    });
  };

  const handleSend = async () => {
    const queryText = input;
    if (!queryText.trim()) return;

    let chatId = selectedChatId;
    if (!chatId || isDraftNewChat) {
      chatId = uid("chat");
      const newSession: ChatSession = {
        id: chatId,
        title: deriveTitle(queryText),
        messages: [],
      };
      setSessions((prev) => ({ ...prev, [chatId!]: newSession }));
      setSelectedChatId(chatId);
      setIsDraftNewChat(false);
    }

    const userMessage: AssistantMessage = {
      id: `msg_${Date.now()}`,
      type: "user",
      content: queryText,
      mode: "chat",
    };
    appendToSession(chatId, userMessage);

    setInput("");
    setIsLoading(true);

    try {
      const response = await sendAssistantQuery(queryText);
      const aiMessage: AssistantMessage = {
        ...response,
        id: response.id ?? `msg_ai_${Date.now()}`,
        type: "ai",
      };
      appendToSession(chatId, aiMessage);
    } catch {
      appendToSession(chatId, {
        id: `msg_err_${Date.now()}`,
        type: "ai",
        content: "I'm sorry, I'm having trouble connecting to the service. Please make sure the backend is running.",
        success: false,
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col bg-background">
      <div className="h-[64px] px-5 border-b border-border bg-card flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-primary/10 flex items-center justify-center">
            <MessageSquare className="w-5 h-5 text-primary" />
          </div>
          <div>
            <div className="text-sm font-bold text-foreground">Manager Assistant</div>
            <div className="text-[11px] text-muted-foreground">New chat and recent chats</div>
          </div>
        </div>
        <button type="button" onClick={createNewChat} className="inline-flex items-center gap-1.5 h-9 px-3 rounded-lg border border-border text-xs font-semibold text-foreground hover:bg-accent">
          <Plus className="w-4 h-4" />
          New chat
        </button>
      </div>

      <div className="flex flex-1 min-h-0">
        <aside className="w-[290px] shrink-0 border-r border-border bg-card/40 hidden lg:flex flex-col min-h-0">
          <div className="h-12 px-3 border-b border-border/70 flex items-center">
            <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Recent chats</span>
          </div>
          <div className="p-2 overflow-y-auto min-h-0 space-y-1">
            {chatList.map((chat) => (
              <div key={chat.id} className="relative">
                <button
                  type="button"
                  onClick={() => openChat(chat.id)}
                  className={`w-full text-left rounded-md px-2.5 py-2 text-xs pr-8 ${selectedChatId === chat.id && !isDraftNewChat ? "bg-primary/15 text-primary" : "text-muted-foreground hover:bg-muted hover:text-foreground"}`}
                >
                  <span className="block truncate">{chat.title}</span>
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    const rect = (e.currentTarget as HTMLButtonElement).getBoundingClientRect();
                    setMenuState((prev) =>
                      prev?.chatId === chat.id
                        ? null
                        : {
                            chatId: chat.id,
                            x: Math.max(8, rect.right - 128),
                            y: rect.bottom + 6,
                          }
                    );
                  }}
                  className="absolute top-1/2 -translate-y-1/2 right-1 w-6 h-6 rounded hover:bg-accent flex items-center justify-center text-muted-foreground"
                >
                  <MoreHorizontal className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        </aside>

        <div className="flex-1 min-w-0 min-h-0 flex flex-col">
          <div className="flex-1 overflow-y-auto p-5 space-y-4 min-h-0">
            {historyError && <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive" role="alert">{historyError}</div>}

            {historyLoading ? (
              <div className="flex flex-col items-center justify-center h-full gap-3" data-cy="assistant-history-loading">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                  <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                  <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
                <p className="text-sm text-muted-foreground font-medium">Loading conversation...</p>
              </div>
            ) : isDraftNewChat ? (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <h2 className="text-2xl font-bold text-foreground mb-1">Start a new chat</h2>
                <p className="text-sm text-muted-foreground">Ask your first question below.</p>
              </div>
            ) : !selectedSession ? (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <h2 className="text-2xl font-bold text-foreground mb-1">No chat selected</h2>
                <p className="text-sm text-muted-foreground">Select a recent chat or click New chat.</p>
              </div>
            ) : visibleMessages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <h2 className="text-2xl font-bold text-foreground mb-1">{selectedSession.title}</h2>
                <p className="text-sm text-muted-foreground">This chat is empty. Ask your first question below.</p>
              </div>
            ) : (
              <>
                {visibleMessages.map((message) => {
                  const timeLabel = formatMessageTime(message.created_at);
                  return (
                    <div key={message.id} className={`flex flex-col gap-1 ${message.type === "user" ? "items-end" : "items-start"}`}>
                      <div className={`flex ${message.type === "user" ? "justify-end" : "justify-start"}`}>
                        {message.type === "user" ? (
                          <div className="max-w-[520px] bg-primary text-primary-foreground rounded-[18px_18px_4px_18px] px-4 py-3"><p className="text-sm">{message.content}</p></div>
                        ) : (
                          <div className="max-w-[92%] md:max-w-[660px] bg-[#1f2937] border border-white/10 rounded-[18px_18px_18px_4px] px-4 py-3 space-y-3 shadow-lg">
                            <p className="text-sm text-white font-medium leading-relaxed">{message.content}</p>
                            {message.success && message.data && message.data.length > 0 && (
                              <div className="border border-white/10 rounded-lg overflow-hidden bg-[#0f172a]"><div className="overflow-x-auto"><table className="min-w-full divide-y divide-white/5"><thead className="bg-[#020617]"><tr>{Object.keys(message.data[0]).map((key) => (<th key={key} className="px-3 py-2 text-left text-[11px] font-bold text-white/50 uppercase tracking-wider">{key.replace(/_/g, " ")}</th>))}</tr></thead><tbody className="bg-[#1f2937] divide-y divide-white/5">{message.data.map((row, idx) => (<tr key={idx} className="hover:bg-white/5 transition-colors">{Object.values(row).map((val: any, vIdx) => (<td key={vIdx} className="px-3 py-2 whitespace-nowrap text-[13px] text-white/90">{typeof val === "number" ? (val % 1 === 0 ? val : val.toFixed(1)) : String(val)}</td>))}</tr>))}</tbody></table></div></div>
                            )}
                            {message.sql && (
                              <details className="cursor-pointer"><summary className="text-[11px] text-white/60 hover:text-white/85 transition-colors mb-1">Show generated SQL</summary><div className="bg-[#0D1117] rounded-lg p-3 overflow-x-auto" style={{ fontFamily: "var(--font-mono)" }}><pre className="text-[10px] text-[#A7F3D0] whitespace-pre-wrap">{message.sql}</pre></div></details>
                            )}
                            {(message.execution_time ?? message.executionTime) && <span className="inline-flex px-2 py-1 bg-white/10 text-white/60 rounded text-[10px] font-medium border border-white/5">Executed in {message.execution_time ?? message.executionTime}</span>}
                          </div>
                        )}
                      </div>
                      {timeLabel && <span className={`text-[10px] text-muted-foreground/80 px-1 ${message.type === "user" ? "text-right" : "text-left"}`}>{timeLabel}</span>}
                    </div>
                  );
                })}
                {isLoading && (
                  <div className="flex justify-start" data-cy="assistant-loading"><div className="bg-card border border-border rounded-[18px_18px_18px_4px] px-4 py-3"><div className="flex gap-1"><div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "0ms" }} /><div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "150ms" }} /><div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "300ms" }} /></div></div></div>
                )}
              </>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="bg-card border-t border-border px-5 py-4 shrink-0">
            <div className="flex items-center gap-2 bg-input border border-border rounded-full p-1 shadow-inner focus-within:ring-1 focus-within:ring-primary/40 transition-all">
              <button type="button" onClick={() => setIsRecording((v) => !v)} className={`w-10 h-10 flex-shrink-0 flex items-center justify-center rounded-xl transition-all ${isRecording ? "bg-destructive/10 text-destructive" : "text-muted-foreground hover:text-foreground"}`}><Mic className="w-5 h-5" /></button>
              <input type="text" value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && handleSend()} placeholder="Ask about scores, violations, agent trends..." disabled={isLoading} className="flex-1 h-10 px-2 bg-transparent text-foreground placeholder-muted-foreground text-sm focus:outline-none disabled:opacity-50" />
              <button type="button" aria-label="Send message" onClick={() => handleSend()} disabled={isLoading || !input.trim()} className={`w-10 h-10 flex-shrink-0 flex items-center justify-center rounded-full transition-all ${input.trim() ? "bg-primary text-primary-foreground shadow-lg shadow-primary/20" : "bg-muted text-muted-foreground opacity-50"}`}><Send className="w-4 h-4" /></button>
            </div>
          </div>
        </div>
      </div>

      {menuState && (
        <div
          className="fixed z-40 w-32 rounded-md border border-border bg-popover shadow-lg p-1"
          style={{ top: menuState.y, left: menuState.x }}
        >
          <button className="w-full text-left text-xs px-2 py-1.5 rounded hover:bg-accent flex items-center gap-1.5" onClick={() => openRenameModal(menuState.chatId)}>
            <Pencil className="w-3.5 h-3.5" /> Rename
          </button>
          <button className="w-full text-left text-xs px-2 py-1.5 rounded hover:bg-destructive/10 text-destructive flex items-center gap-1.5" onClick={() => openDeleteModal(menuState.chatId)}>
            <Trash2 className="w-3.5 h-3.5" /> Delete
          </button>
        </div>
      )}

      {modalState && (
        <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-[1px] flex items-center justify-center p-4">
          <div className="w-full max-w-sm rounded-xl border border-border bg-card shadow-2xl p-4 space-y-3">
            {modalState.type === "rename" ? (
              <>
                <h3 className="text-sm font-semibold text-foreground">Rename chat</h3>
                <input value={renameValue} onChange={(e) => setRenameValue(e.target.value)} className="w-full h-10 px-3 rounded-lg border border-border bg-background text-sm focus:outline-none focus:ring-1 focus:ring-primary/40" autoFocus />
                <div className="flex justify-end gap-2"><button className="h-9 px-3 text-xs rounded-lg border border-border hover:bg-accent" onClick={() => setModalState(null)}>Cancel</button><button className="h-9 px-3 text-xs rounded-lg bg-primary text-primary-foreground hover:opacity-90" onClick={confirmRename}>Save</button></div>
              </>
            ) : (
              <>
                <h3 className="text-sm font-semibold text-foreground">Delete chat?</h3>
                <p className="text-xs text-muted-foreground">Are you sure you want to delete this chat?</p>
                <div className="flex justify-end gap-2"><button className="h-9 px-3 text-xs rounded-lg border border-border hover:bg-accent" onClick={() => setModalState(null)}>Cancel</button><button className="h-9 px-3 text-xs rounded-lg bg-destructive text-destructive-foreground hover:opacity-90" onClick={confirmDelete}>Delete</button></div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
