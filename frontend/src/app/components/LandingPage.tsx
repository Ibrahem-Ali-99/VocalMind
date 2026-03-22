import { Link } from "react-router";
import { Mic, UserCircle, Shield } from "lucide-react";

export function LandingPage() {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-6">
      <div className="max-w-5xl w-full">
        {/* Logo and Title */}
        <div className="text-center mb-16">
          <div className="flex items-center justify-center gap-4 mb-6">
            <div className="w-16 h-16 bg-primary rounded-xl flex items-center justify-center shadow-lg shadow-primary/20">
              <Mic className="w-8 h-8 text-primary-foreground" />
            </div>
            <h1 className="text-5xl font-bold text-foreground" style={{ fontFamily: 'var(--font-sans)' }}>
              VocalMind
            </h1>
          </div>
          <p className="text-xl text-muted-foreground">
            AI-Powered Call Centre Evaluation Platform
          </p>
        </div>

        {/* Role Selection */}
        <div className="grid md:grid-cols-2 gap-8">
          {/* Manager Portal */}
          <Link
            to="/manager"
            className="group bg-card rounded-2xl p-8 border border-border hover:border-primary transition-all duration-300 hover:shadow-xl active:scale-[0.98]"
          >
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 bg-primary/10 rounded-xl flex items-center justify-center group-hover:bg-primary transition-colors">
                <Shield className="w-6 h-6 text-primary group-hover:text-primary-foreground transition-colors" />
              </div>
              <div>
                <h2 className="text-2xl font-semibold text-foreground mb-1">
                  Manager Portal
                </h2>
                <p className="text-sm text-primary font-medium">
                  Full org access
                </p>
              </div>
            </div>
            
            <div className="space-y-3 mb-6">
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 bg-[#3B82F6] rounded-full mt-2" />
                <p className="text-sm text-[#D1D5DB]">
                  Comprehensive dashboard with org-wide KPIs
                </p>
              </div>
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 bg-[#3B82F6] rounded-full mt-2" />
                <p className="text-sm text-[#D1D5DB]">
                  Session Inspector with emotion detection
                </p>
              </div>
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 bg-[#3B82F6] rounded-full mt-2" />
                <p className="text-sm text-[#D1D5DB]">
                  AI Assistant for data queries
                </p>
              </div>
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 bg-[#3B82F6] rounded-full mt-2" />
                <p className="text-sm text-[#D1D5DB]">
                  Knowledge Base management
                </p>
              </div>
            </div>

            <div className="text-primary font-bold group-hover:translate-x-1 transition-transform inline-flex items-center gap-2">
              Enter Manager Portal →
            </div>
          </Link>

          {/* Agent Portal */}
          <Link
            to="/agent"
            className="group bg-card rounded-2xl p-8 border border-border hover:border-success transition-all duration-300 hover:shadow-xl active:scale-[0.98]"
          >
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 bg-success/10 rounded-xl flex items-center justify-center group-hover:bg-success transition-colors">
                <UserCircle className="w-6 h-6 text-success group-hover:text-primary-foreground transition-colors" />
              </div>
              <div>
                <h2 className="text-2xl font-semibold text-foreground mb-1">
                  Agent Portal
                </h2>
                <p className="text-sm text-success font-medium">
                  Personal view only
                </p>
              </div>
            </div>
            
            <div className="space-y-3 mb-6">
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 bg-[#10B981] rounded-full mt-2" />
                <p className="text-sm text-[#D1D5DB]">
                  Personal performance metrics and trends
                </p>
              </div>
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 bg-[#10B981] rounded-full mt-2" />
                <p className="text-sm text-[#D1D5DB]">
                  Individual call transcripts and analysis
                </p>
              </div>
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 bg-[#10B981] rounded-full mt-2" />
                <p className="text-sm text-[#D1D5DB]">
                  Coaching points for improvement
                </p>
              </div>
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 bg-[#10B981] rounded-full mt-2" />
                <p className="text-sm text-[#D1D5DB]">
                  Customer emotion journey insights
                </p>
              </div>
            </div>

            <div className="text-success font-bold group-hover:translate-x-1 transition-transform inline-flex items-center gap-2">
              Enter Agent Portal →
            </div>
          </Link>
        </div>

        {/* Footer Note */}
        <div className="text-center mt-12 text-muted-foreground text-sm">
          <p>Each portal provides a tailored experience based on your role</p>
        </div>
      </div>
    </div>
  );
}
