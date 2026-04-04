import React, { useState } from "react";
import { Navigate, useNavigate } from "react-router";
import { useAuth } from "../contexts/AuthContext";
import { Headphones, Mail, Lock, LogIn, AlertCircle } from "lucide-react";

export default function Login() {
  const { login, user, isAuthenticated, isLoading: authLoading } = useAuth();
  const navigate = useNavigate();
  
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  // If we are still checking if the user is logged in, show a blank state or spinner
  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="w-8 h-8 border-4 border-primary/30 border-t-primary rounded-full animate-spin" />
      </div>
    );
  }

  // If already logged in, redirect away
  if (isAuthenticated) {
    return <Navigate to={user?.role === "agent" ? "/agent" : "/manager"} replace />;
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      const user = await login(email, password);
      if (user?.role === "agent") {
        navigate("/agent");
      } else {
        navigate("/manager");
      }
    } catch (err: any) {
      setError(err.message || "Failed to sign in. Please check your credentials.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background relative overflow-hidden">
      {/* Background purely for aesthetic glassmorphism feel mapping */}
      <div className="absolute inset-0 z-0 pointer-events-none">
        <div className="absolute -top-[10%] -right-[10%] w-[50%] h-[50%] rounded-full bg-primary/10 blur-[100px] opacity-60" />
        <div className="absolute -bottom-[10%] -left-[10%] w-[50%] h-[50%] rounded-full bg-accent/10 blur-[100px] opacity-60" />
      </div>

      <div className="relative z-10 w-full max-w-[400px] px-6">
        <div className="flex flex-col items-center mb-8">
          <div className="w-12 h-12 bg-gradient-to-br from-[#1D4ED8] to-[#3B82F6] rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20 mb-4">
            <Headphones className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-[24px] font-bold text-foreground tracking-tight">Welcome to VocalMind</h1>
          <p className="text-[14px] text-muted-foreground mt-1">Sign in to your dashboard</p>
        </div>

        <div className="bg-card/80 backdrop-blur-xl border border-border rounded-2xl shadow-xl shadow-black/5 p-8">
          <form onSubmit={handleSubmit} className="flex flex-col gap-5">
            {error && (
              <div className="flex items-center gap-2 p-3 bg-destructive/10 border border-destructive/20 rounded-lg text-destructive text-[13px]">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <p>{error}</p>
              </div>
            )}

            <div>
              <label className="block text-[13px] font-medium text-foreground mb-1.5">
                Work Email
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Mail className="h-4 w-4 text-muted-foreground" />
                </div>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="block w-full pl-10 pr-3 py-2.5 border border-border rounded-xl text-[14px] focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all bg-background placeholder-muted-foreground outline-none"
                  placeholder="employee@vocalmind.ai"
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-[13px] font-medium text-foreground mb-1.5">
                Password
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock className="h-4 w-4 text-muted-foreground" />
                </div>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="block w-full pl-10 pr-3 py-2.5 border border-border rounded-xl text-[14px] focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all bg-background placeholder-muted-foreground outline-none"
                  placeholder="••••••••"
                  required
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full flex items-center justify-center gap-2 bg-primary hover:bg-primary/90 text-primary-foreground py-2.5 rounded-xl text-[14px] font-medium transition-colors shadow-sm disabled:opacity-70 disabled:cursor-not-allowed mt-2"
            >
              {isLoading ? (
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <>
                  <LogIn className="w-4 h-4" />
                  Sign In
                </>
              )}
            </button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-[12px] text-[#6B7280]">
              Please contact your administrator if you cannot access your account.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
