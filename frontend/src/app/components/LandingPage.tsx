import { Link } from "react-router";
import { Mic, UserCircle, Shield } from "lucide-react";

export function LandingPage() {
  return (
    <div className="min-h-screen bg-[#0D1117] flex items-center justify-center p-6">
      <div className="max-w-5xl w-full">
        {/* Logo and Title */}
        <div className="text-center mb-16">
          <div className="flex items-center justify-center gap-4 mb-6">
            <div className="w-16 h-16 bg-[#3B82F6] rounded-xl flex items-center justify-center">
              <Mic className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-5xl font-bold text-white" style={{ fontFamily: 'var(--font-sans)' }}>
              VocalMind
            </h1>
          </div>
          <p className="text-xl text-[#9CA3AF]">
            AI-Powered Call Centre Evaluation Platform
          </p>
        </div>

        {/* Role Selection */}
        <div className="grid md:grid-cols-2 gap-8">
          {/* Manager Portal */}
          <Link
            to="/manager"
            className="group bg-[#1F2937] rounded-2xl p-8 border border-[#374151] hover:border-[#3B82F6] transition-all duration-300 hover:shadow-xl"
          >
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 bg-[#EFF6FF] rounded-xl flex items-center justify-center group-hover:bg-[#3B82F6] transition-colors">
                <Shield className="w-6 h-6 text-[#3B82F6] group-hover:text-white transition-colors" />
              </div>
              <div>
                <h2 className="text-2xl font-semibold text-white mb-1">
                  Manager Portal
                </h2>
                <p className="text-sm text-[#93C5FD]">
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

            <div className="text-[#3B82F6] font-semibold group-hover:translate-x-1 transition-transform inline-flex items-center gap-2">
              Enter Manager Portal →
            </div>
          </Link>

          {/* Agent Portal */}
          <Link
            to="/agent"
            className="group bg-[#1F2937] rounded-2xl p-8 border border-[#374151] hover:border-[#10B981] transition-all duration-300 hover:shadow-xl"
          >
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 bg-[#ECFDF5] rounded-xl flex items-center justify-center group-hover:bg-[#10B981] transition-colors">
                <UserCircle className="w-6 h-6 text-[#10B981] group-hover:text-white transition-colors" />
              </div>
              <div>
                <h2 className="text-2xl font-semibold text-white mb-1">
                  Agent Portal
                </h2>
                <p className="text-sm text-[#6EE7B7]">
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

            <div className="text-[#10B981] font-semibold group-hover:translate-x-1 transition-transform inline-flex items-center gap-2">
              Enter Agent Portal →
            </div>
          </Link>
        </div>

        {/* Footer Note */}
        <div className="text-center mt-12 text-[#6B7280] text-sm">
          <p>Each portal provides a tailored experience based on your role</p>
        </div>
      </div>
    </div>
  );
}
