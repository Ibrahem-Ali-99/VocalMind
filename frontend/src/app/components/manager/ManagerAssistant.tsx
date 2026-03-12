import { useState } from "react";
import { MessageSquare, Mic, Send } from "lucide-react";

interface AssistantMessage {
  id: string;
  type: "user" | "ai";
  content: string;
  mode?: string;
  sql?: string;
  executionTime?: string;
}

export function ManagerAssistant() {
  const [messages, setMessages] = useState<AssistantMessage[]>([]);
  const [input, setInput] = useState("");
  const [isRecording, setIsRecording] = useState(false);

  const suggestedQueries = [
    "Show top performing agents this week",
    "List all policy violations today",
    "Which agent has the lowest resolution rate?",
    "Show emotion trends across all calls",
  ];

  const handleSend = () => {
    if (!input.trim()) return;

    const newUserMessage = {
      id: `msg_${Date.now()}`,
      type: "user" as const,
      mode: "chat" as const,
      content: input,
    };

    const newAiMessage = {
      id: `msg_${Date.now() + 1}`,
      type: "ai" as const,
      content: "I've analyzed your query. Here are the results based on the current data in your organization's database.",
      sql: `SELECT * FROM interactions WHERE organization_id = $1 ORDER BY created_at DESC LIMIT 10`,
      executionTime: "98ms",
    };

    setMessages([...messages, newUserMessage, newAiMessage]);
    setInput("");
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div
        className="h-[72px] px-6 flex items-center justify-between"
        style={{ background: "linear-gradient(135deg, #2563EB 0%, #3B82F6 100%)" }}
      >
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center">
            <MessageSquare className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-[16px] font-bold text-white">
              Manager Assistant
            </h2>
            <p className="text-[12px] text-white/70">
              Ask anything about your call centre · voice or text
            </p>
          </div>
        </div>

        <button
          onClick={() => setIsRecording(!isRecording)}
          className={`w-10 h-10 rounded-lg flex items-center justify-center transition-all ${
            isRecording
              ? "bg-[#FEF2F2] animate-pulse"
              : "bg-white/20 hover:bg-white/30"
          }`}
        >
          <Mic className={`w-5 h-5 ${isRecording ? "text-[#EF4444]" : "text-white"}`} />
        </button>
      </div>

      {/* Messages Area */}
      <div className="flex-1 bg-[#F9FAFB] overflow-y-auto p-5 space-y-3">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full">
            <div className="w-14 h-14 bg-[#EFF6FF] rounded-2xl flex items-center justify-center mb-4">
              <MessageSquare className="w-7 h-7 text-[#3B82F6]" />
            </div>
            <h3 className="text-[18px] font-bold text-[#374151] mb-2">
              Ask anything about your call centre
            </h3>
            <p className="text-[13px] text-[#9CA3AF]">
              Voice or text — queries are logged to assistant_queries
            </p>
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.type === "user" ? "justify-end" : "justify-start"}`}
              >
                {message.type === "user" ? (
                  <div className="max-w-[480px] bg-[#3B82F6] text-white rounded-[18px_18px_4px_18px] px-4 py-3">
                    <p className="text-[14px]">{message.content}</p>
                  </div>
                ) : (
                  <div className="max-w-[520px] bg-white border border-[#E5E7EB] shadow-sm rounded-[18px_18px_18px_4px] px-4 py-3 space-y-3">
                    <p className="text-[14px] text-[#374151]">{message.content}</p>
                    
                    {message.sql && (
                      <div
                        className="bg-[#0D1117] rounded-lg p-3 overflow-x-auto"
                        style={{ fontFamily: 'var(--font-mono)' }}
                      >
                        <pre className="text-[11px] text-[#A7F3D0] whitespace-pre-wrap">
                          {message.sql}
                        </pre>
                      </div>
                    )}

                    {message.executionTime && (
                      <div className="flex items-center gap-2">
                        <span className="px-2 py-1 bg-[#F3F4F6] text-[#9CA3AF] rounded text-[10px]">
                          Executed in {message.executionTime}
                        </span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </>
        )}
      </div>

      {/* Suggested Queries */}
      <div className="bg-white border-t border-[#E5E7EB] px-5 py-3">
        <div className="text-[11px] font-semibold uppercase tracking-wide text-[#9CA3AF] mb-2">
          Suggested queries
        </div>
        <div className="grid grid-cols-2 gap-2">
          {suggestedQueries.map((query, index) => (
            <button
              key={index}
              onClick={() => setInput(query)}
              className="px-3 py-2 bg-[#EFF6FF] text-[#1D4ED8] border border-[#BFDBFE] rounded-lg text-[12px] text-left hover:bg-[#DBEAFE] transition-colors"
            >
              {query}
            </button>
          ))}
        </div>
      </div>

      {/* Input Area */}
      <div className="bg-white border-t border-[#E5E7EB] px-5 py-4">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsRecording(!isRecording)}
            className={`w-11 h-11 flex-shrink-0 flex items-center justify-center rounded-lg border transition-all ${
              isRecording
                ? "bg-[#FEF2F2] border-[#FEE2E2]"
                : "bg-[#F3F4F6] border-[#E5E7EB] hover:bg-[#E5E7EB]"
            }`}
          >
            <Mic className={`w-5 h-5 ${isRecording ? "text-[#EF4444]" : "text-[#6B7280]"}`} />
          </button>

          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder="Ask about scores, violations, agent trends…"
            className="flex-1 h-11 px-4 bg-[#F9FAFB] border border-[#E5E7EB] rounded-xl text-[14px] focus:outline-none focus:ring-2 focus:ring-[#3B82F6]"
          />

          <button
            onClick={handleSend}
            className="w-11 h-11 flex-shrink-0 bg-[#2563EB] hover:bg-[#1D4ED8] flex items-center justify-center rounded-xl transition-colors"
          >
            <Send className="w-5 h-5 text-white" />
          </button>
        </div>
      </div>
    </div>
  );
}
