import { useState, useEffect, useRef } from "react";
import { MessageSquare, Mic, Send } from "lucide-react";
import { sendAssistantQuery, getAssistantHistory, AssistantResponse } from "../../services/api";

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
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getAssistantHistory()
      .then((history) => {
        if (history && history.length > 0) {
          setMessages(history as AssistantMessage[]);
        }
      })
      .catch((err) => console.error("Failed to load chat history:", err));
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const suggestedQueries = [
    "Who are the top 5 agents by overall score?",
    "List all policy violations",
    "What are the most common customer emotions?",
    "help",
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
      <div className="h-[64px] px-6 flex items-center justify-between bg-card border-b-2 border-primary/40 shadow-sm">
        <div className="flex items-center gap-4">
          <div className="w-9 h-9 bg-primary/10 rounded-xl flex items-center justify-center">
            <MessageSquare className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h2 className="text-[15px] font-bold text-foreground">
              Manager Assistant
            </h2>
            <p className="text-[11px] text-muted-foreground font-medium">
              Enterprise Voice & Text Analysis
            </p>
          </div>
        </div>

        <button
          onClick={() => setIsRecording(!isRecording)}
          className={`w-9 h-9 rounded-lg flex items-center justify-center transition-all ${
            isRecording
              ? "bg-destructive/10 animate-pulse border border-destructive/20"
              : "bg-muted hover:bg-muted/80 text-muted-foreground"
          }`}
        >
          <Mic className={`w-4 h-4 ${isRecording ? "text-destructive" : "currentColor"}`} />
        </button>
      </div>

      {/* Messages Area */}
      <div className="flex-1 bg-background overflow-y-auto p-5 space-y-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full">
            <div className="w-14 h-14 bg-primary/10 rounded-2xl flex items-center justify-center mb-4">
              <MessageSquare className="w-7 h-7 text-primary" />
            </div>
            <h3 className="text-[18px] font-bold text-foreground mb-2">
              Ask anything about your call centre
            </h3>
            <p className="text-[13px] text-muted-foreground font-medium">
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
                  <div className="max-w-[480px] bg-primary text-primary-foreground rounded-[18px_18px_4px_18px] px-4 py-3">
                    <p className="text-[14px]">{message.content}</p>
                  </div>
                ) : (
                  <div className="max-w-[90%] md:max-w-[600px] bg-[#1f2937] border border-white/5 rounded-[18px_18px_18px_4px] px-4 py-3 space-y-3 shadow-lg">
                    <p className="text-[14px] text-white font-medium leading-relaxed">{message.content}</p>
                    
                    {/* Render Data Table if success and data exists */}
                    {message.success && message.data && message.data.length > 0 && (
                      <div className="border border-white/10 rounded-lg overflow-hidden bg-[#0f172a] shadow-inner">
                        <div className="overflow-x-auto">
                          <table className="min-w-full divide-y divide-white/5">
                            <thead className="bg-[#020617]">
                              <tr>
                                {Object.keys(message.data[0]).map((key) => (
                                  <th
                                    key={key}
                                    className="px-3 py-2 text-left text-[11px] font-bold text-white/50 uppercase tracking-wider"
                                  >
                                    {key.replace(/_/g, " ")}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody className="bg-[#1f2937] divide-y divide-white/5">
                              {message.data.map((row, idx) => (
                                <tr key={idx} className="hover:bg-white/5 transition-colors">
                                  {Object.values(row).map((val: any, vIdx) => (
                                    <td key={vIdx} className="px-3 py-2 whitespace-nowrap text-[13px] text-white/90">
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
                        <summary className="text-[11px] text-white/50 hover:text-white/80 transition-colors mb-1">
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
                        <span className="px-2 py-1 bg-white/10 text-white/60 rounded text-[10px] font-medium border border-white/5">
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
                  <div className="bg-card border border-border shadow-sm rounded-[18px_18px_18px_4px] px-4 py-3">
                     <div className="flex gap-1">
                        <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                        <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                        <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                     </div>
                  </div>
               </div>
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Queries */}
      <div className="bg-card border-t border-border px-5 py-3">
        <div className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground mb-2">
          Suggested queries
        </div>
        <div className="grid grid-cols-2 gap-2">
          {suggestedQueries.map((query, index) => (
            <button
              key={index}
              onClick={() => handleSend(query)}
              className="px-3 py-2 bg-primary/10 text-primary border border-primary/20 rounded-lg text-[12px] text-left hover:bg-primary/20 transition-colors disabled:opacity-50"
              disabled={isLoading}
            >
              {query}
            </button>
          ))}
        </div>
      </div>

      {/* Input Area */}
      <div className="bg-card border-t border-border px-5 py-4">
        <div className="flex items-center gap-2 bg-input border border-border rounded-full p-1 shadow-inner focus-within:ring-1 focus-within:ring-primary/40 transition-all">
          <button
            onClick={() => setIsRecording(!isRecording)}
            className={`w-10 h-10 flex-shrink-0 flex items-center justify-center rounded-xl transition-all ${
              isRecording
                ? "bg-destructive/10 text-destructive"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Mic className="w-5 h-5" />
          </button>

          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder="Ask about scores, violations, agent trends…"
            disabled={isLoading}
            className="flex-1 h-10 px-2 bg-transparent text-foreground placeholder-muted-foreground text-[14px] focus:outline-none disabled:opacity-50"
          />

          <button
            onClick={() => handleSend()}
            disabled={isLoading || !input.trim()}
            className={`w-10 h-10 flex-shrink-0 flex items-center justify-center rounded-full transition-all ${
              input.trim() 
                ? "bg-primary text-primary-foreground shadow-lg shadow-primary/20" 
                : "bg-muted text-muted-foreground opacity-50"
            }`}
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
