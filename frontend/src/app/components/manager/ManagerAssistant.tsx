import { useState } from "react";
import { MessageSquare, Mic, Send } from "lucide-react";
import { sendAssistantQuery, AssistantResponse } from "../../services/api";

interface AssistantMessage extends Partial<AssistantResponse> {
  id: string;
  type: "user" | "ai";
  content: string;
  mode?: string;
}

export function ManagerAssistant() {
  const [messages, setMessages] = useState<AssistantMessage[]>([]);
  const [input, setInput] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const suggestedQueries = [
    "Show top performing agents this week",
    "List all policy violations today",
    "Which agent has the lowest resolution rate?",
    "Show emotion trends across all calls",
  ];

  const handleSend = async (textOverride?: string) => {
    const queryText = textOverride || input;
    if (!queryText.trim()) return;

    const userMsgId = `msg_${Date.now()}`;
    const newUserMessage: AssistantMessage = {
      id: userMsgId,
      type: "user",
      content: queryText,
    };

    setMessages((prev) => [...prev, newUserMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await sendAssistantQuery(queryText);
      const aiMessage: AssistantMessage = {
        ...response,
        type: "ai",
      };
      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      const errorMessage: AssistantMessage = {
        id: `msg_err_${Date.now()}`,
        type: "ai",
        content: "I'm sorry, I'm having trouble connecting to the service. Please make sure the backend is running.",
        success: false
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
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
      <div className="flex-1 bg-[#F9FAFB] overflow-y-auto p-5 space-y-4">
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
                  <div className="max-w-[90%] md:max-w-[600px] bg-white border border-[#E5E7EB] shadow-sm rounded-[18px_18px_18px_4px] px-4 py-3 space-y-3">
                    <p className="text-[14px] text-[#374151] font-medium">{message.content}</p>
                    
                    {/* Render Data Table if success and data exists */}
                    {message.success && message.data && message.data.length > 0 && (
                      <div className="border border-[#F3F4F6] rounded-lg overflow-hidden bg-white">
                        <div className="overflow-x-auto">
                          <table className="min-w-full divide-y divide-[#F3F4F6]">
                            <thead className="bg-[#F9FAFB]">
                              <tr>
                                {Object.keys(message.data[0]).map((key) => (
                                  <th
                                    key={key}
                                    className="px-3 py-2 text-left text-[11px] font-bold text-[#6B7280] uppercase tracking-wider"
                                  >
                                    {key.replace(/_/g, " ")}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-[#F3F4F6]">
                              {message.data.map((row, idx) => (
                                <tr key={idx} className="hover:bg-[#F9FAFB]">
                                  {Object.values(row).map((val: any, vIdx) => (
                                    <td key={vIdx} className="px-3 py-2 whitespace-nowrap text-[13px] text-[#374151]">
                                      {typeof val === "number" ? (val % 1 === 0 ? val : val.toFixed(1)) : String(val)}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}

                    {message.sql && (
                      <details className="cursor-pointer">
                        <summary className="text-[11px] text-[#9CA3AF] hover:text-[#6B7280] transition-colors mb-1">
                          Show generated SQL
                        </summary>
                        <div
                          className="bg-[#0D1117] rounded-lg p-3 overflow-x-auto"
                          style={{ fontFamily: 'var(--font-mono)' }}
                        >
                          <pre className="text-[10px] text-[#A7F3D0] whitespace-pre-wrap">
                            {message.sql}
                          </pre>
                        </div>
                      </details>
                    )}

                    {message.execution_time && (
                      <div className="flex items-center gap-2">
                        <span className="px-2 py-1 bg-[#F3F4F6] text-[#9CA3AF] rounded text-[10px]">
                          Executed in {message.execution_time}
                        </span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
            {isLoading && (
               <div className="flex justify-start">
                  <div className="bg-white border border-[#E5E7EB] shadow-sm rounded-[18px_18px_18px_4px] px-4 py-3">
                     <div className="flex gap-1">
                        <div className="w-2 h-2 bg-[#3B82F6] rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                        <div className="w-2 h-2 bg-[#3B82F6] rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                        <div className="w-2 h-2 bg-[#3B82F6] rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                     </div>
                  </div>
               </div>
            )}
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
              onClick={() => handleSend(query)}
              className="px-3 py-2 bg-[#EFF6FF] text-[#1D4ED8] border border-[#BFDBFE] rounded-lg text-[12px] text-left hover:bg-[#DBEAFE] transition-colors disabled:opacity-50"
              disabled={isLoading}
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
            disabled={isLoading}
            className="flex-1 h-11 px-4 bg-[#F9FAFB] border border-[#E5E7EB] rounded-xl text-[14px] focus:outline-none focus:ring-2 focus:ring-[#3B82F6] disabled:opacity-50"
          />

          <button
            onClick={() => handleSend()}
            disabled={isLoading || !input.trim()}
            className="w-11 h-11 flex-shrink-0 bg-[#2563EB] hover:bg-[#1D4ED8] disabled:bg-gray-400 flex items-center justify-center rounded-xl transition-colors"
          >
            <Send className="w-5 h-5 text-white" />
          </button>
        </div>
      </div>
    </div>
  );
}
