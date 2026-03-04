import { Outlet, Link, useLocation } from "react-router";
import { useState } from "react";
import {
  Mic,
  Activity,
  Phone,
  Settings,
  ChevronLeft,
  ChevronRight,
  Bell,
} from "lucide-react";

export function AgentLayout() {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();

  const navItems = [
    { icon: Activity, label: "My Performance", path: "/agent" },
    { icon: Phone, label: "My Calls", path: "/agent/calls" },
    { icon: Settings, label: "Settings", path: "/agent/settings" },
  ];

  const getPageTitle = () => {
    if (location.pathname === "/agent") return "My Performance";
    if (location.pathname.includes("calls")) return "Call Detail";
    return "My Performance";
  };

  return (
    <div className="flex h-screen bg-[#F3F4F6]">
      {/* Sidebar */}
      <div
        className={`${
          collapsed ? "w-[72px]" : "w-[240px]"
        } bg-[#0D1117] border-r border-[#1F2937] flex flex-col transition-all duration-300`}
      >
        {/* Logo Area */}
        <div className="h-16 flex items-center px-4 border-b border-[#1F2937]">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-[#10B981] rounded-xl flex items-center justify-center flex-shrink-0">
              <Mic className="w-[18px] h-[18px] text-white" />
            </div>
            {!collapsed && (
              <span className="text-white font-bold text-lg" style={{ fontFamily: 'var(--font-sans)' }}>
                VocalMind
              </span>
            )}
          </div>
        </div>

        {/* Role Badge */}
        {!collapsed && (
          <div className="px-4 py-4">
            <div className="bg-[#062014] border border-[#065F46] rounded-lg p-3">
              <div className="text-[13px] font-semibold text-[#6EE7B7] mb-0.5">
                Agent Portal
              </div>
              <div className="text-[11px] text-[#6B7280]">
                Personal view only
              </div>
            </div>
          </div>
        )}

        {/* Navigation */}
        <nav className="flex-1 px-2 py-2 space-y-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path || 
              (item.path === "/agent/calls" && location.pathname.includes("/agent/calls"));
            
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 px-3 h-11 rounded-lg transition-all ${
                  isActive
                    ? "bg-[#10B981] text-white shadow-lg shadow-[#10B981]/20"
                    : "text-[#6B7280] hover:text-[#E5E7EB] hover:bg-[#1F2937]"
                }`}
              >
                <Icon className="w-[18px] h-[18px] flex-shrink-0" />
                {!collapsed && (
                  <span className="text-[14px] font-medium">
                    {item.label}
                  </span>
                )}
              </Link>
            );
          })}
        </nav>

        {/* Bottom Section */}
        <div className="border-t border-[#1F2937] p-4">
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="w-full flex items-center justify-center h-8 text-[#6B7280] hover:text-white transition-colors mb-3"
          >
            {collapsed ? (
              <ChevronRight className="w-5 h-5" />
            ) : (
              <ChevronLeft className="w-5 h-5" />
            )}
          </button>

          {!collapsed && (
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-[#10B981] flex items-center justify-center text-white text-xs font-semibold">
                RK
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-white text-xs font-medium truncate">
                  Rajesh Kumar
                </div>
                <div className="text-[#6B7280] text-[11px]">
                  Agent
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Bar */}
        <div className="h-14 bg-white border-b border-[#D1FAE5] px-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-[16px] font-bold text-[#111827]">
              {getPageTitle()}
            </h1>
            <span className="px-2.5 py-1 bg-[#ECFDF5] text-[#10B981] border border-[#A7F3D0] rounded-full text-[11px] font-semibold uppercase tracking-wide">
              Agent
            </span>
          </div>

          <div className="flex items-center gap-3">
            <button className="w-8 h-8 flex items-center justify-center bg-[#F9FAFB] border border-[#E5E7EB] rounded-lg hover:bg-[#F3F4F6] transition-colors">
              <Bell className="w-4 h-4 text-[#6B7280]" />
            </button>
            <div className="w-8 h-8 rounded-full bg-[#10B981] flex items-center justify-center text-white text-xs font-semibold">
              RK
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          <Outlet />
        </div>
      </div>
    </div>
  );
}
