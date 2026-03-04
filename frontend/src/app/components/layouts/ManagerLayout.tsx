import { Outlet, Link, useLocation } from "react-router";
import { useState } from "react";
import {
  Mic,
  LayoutDashboard,
  Search,
  MessageSquare,
  BookOpen,
  Settings,
  ChevronLeft,
  ChevronRight,
  Bell,
  Download,
} from "lucide-react";

export function ManagerLayout() {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();

  const navItems = [
    { icon: LayoutDashboard, label: "Dashboard", path: "/manager" },
    { icon: Search, label: "Session Inspector", path: "/manager/inspector" },
    { icon: MessageSquare, label: "Manager Assistant", path: "/manager/assistant" },
    { icon: BookOpen, label: "Knowledge Base", path: "/manager/knowledge" },
    { icon: Settings, label: "Settings", path: "/manager/settings" },
  ];

  const getPageTitle = () => {
    if (location.pathname === "/manager") return "Dashboard";
    if (location.pathname.includes("inspector") && !location.pathname.includes("/manager/inspector/")) return "Session Inspector";
    if (location.pathname.includes("inspector/")) return "Call Detail";
    if (location.pathname.includes("assistant")) return "Manager Assistant";
    if (location.pathname.includes("knowledge")) return "Knowledge Base";
    return "Dashboard";
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
            <div className="w-9 h-9 bg-[#3B82F6] rounded-xl flex items-center justify-center flex-shrink-0">
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
            <div className="bg-[#0C1A3A] border border-[#1D3A6E] rounded-lg p-3">
              <div className="text-[13px] font-semibold text-[#93C5FD] mb-0.5">
                Manager Portal
              </div>
              <div className="text-[11px] text-[#6B7280]">
                Full org access
              </div>
            </div>
          </div>
        )}

        {/* Navigation */}
        <nav className="flex-1 px-2 py-2 space-y-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path || 
              (item.path === "/manager/inspector" && location.pathname.includes("/manager/inspector"));
            
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 px-3 h-11 rounded-lg transition-all ${
                  isActive
                    ? "bg-[#3B82F6] text-white shadow-lg shadow-[#3B82F6]/20"
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
              <div className="w-8 h-8 rounded-full bg-[#3B82F6] flex items-center justify-center text-white text-xs font-semibold">
                MK
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-white text-xs font-medium truncate">
                  Manager User
                </div>
                <div className="text-[#6B7280] text-[11px]">
                  Manager
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Bar */}
        <div className="h-14 bg-white border-b border-[#DBEAFE] px-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-[16px] font-bold text-[#111827]">
              {getPageTitle()}
            </h1>
            <span className="px-2.5 py-1 bg-[#EFF6FF] text-[#3B82F6] border border-[#BFDBFE] rounded-full text-[11px] font-semibold uppercase tracking-wide">
              Manager
            </span>
          </div>

          <div className="flex items-center gap-3">
            <button className="flex items-center gap-2 px-3 h-8 border border-[#E5E7EB] rounded-lg text-[13px] text-[#6B7280] hover:bg-[#F9FAFB] transition-colors">
              <Download className="w-3.5 h-3.5" />
              Export
            </button>
            <button className="w-8 h-8 flex items-center justify-center bg-[#F9FAFB] border border-[#E5E7EB] rounded-lg hover:bg-[#F3F4F6] transition-colors">
              <Bell className="w-4 h-4 text-[#6B7280]" />
            </button>
            <div className="w-8 h-8 rounded-full bg-[#3B82F6] flex items-center justify-center text-white text-xs font-semibold">
              MK
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
