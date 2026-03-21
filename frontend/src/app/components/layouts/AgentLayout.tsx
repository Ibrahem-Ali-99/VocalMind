import { Outlet, Link, useLocation } from "react-router";
import { useState } from "react";
import {
  Menu,
  Activity,
  Phone,
  Settings,
  Bell,
} from "lucide-react";
import logoSrc from "../../../assets/logo/logo.svg";

export function AgentLayout() {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();

  const navItems = [
    { icon: Activity, label: "My Performance", path: "/agent" },
    { icon: Phone, label: "My Calls", path: "/agent/calls" },
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
          collapsed ? "w-[80px]" : "w-[248px]"
        } bg-[#0D1117] border-r border-[#1F2937] flex flex-col transition-all duration-300`}
      >
        {/* Logo + Menu Header */}
        <div className={`border-b border-[#1F2937] ${collapsed ? "px-2 py-4" : "px-4 py-4"}`}>
          <Link
            to="/agent"
            className={`flex items-center ${collapsed ? "justify-center" : "gap-3"} cursor-pointer rounded-xl px-2 py-1.5 hover:bg-[#111827] transition-colors`}
            title="Go to dashboard"
          >
            <img src={logoSrc} alt="VocalMind" className="w-10 h-10 rounded-xl object-contain flex-shrink-0" />
            {!collapsed && (
              <span className="text-white font-semibold text-[18px] leading-none" style={{ fontFamily: "var(--font-sans)" }}>
                VocalMind
              </span>
            )}
          </Link>
          <div className="mt-3 flex justify-center">
            <button
              onClick={() => setCollapsed(!collapsed)}
              className={`${collapsed ? "w-10" : "w-full"} h-9 flex items-center justify-center gap-2 text-[#9CA3AF] hover:text-white bg-[#111827] hover:bg-[#1F2937] border border-[#1F2937] rounded-xl transition-colors`}
              title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              <Menu className="w-5 h-5" />
              {!collapsed && <span className="text-[12px] font-medium">Collapse</span>}
            </button>
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
        <nav className="flex-1 px-2 py-3 space-y-1.5">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path || 
              (item.path === "/agent/calls" && location.pathname.includes("/agent/calls"));
            
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center ${collapsed ? "justify-center" : "gap-3"} px-3 h-11 rounded-xl transition-all ${
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

        {/* Bottom Settings */}
        <div className="border-t border-[#1F2937] p-2">
          <Link
            to="/agent/settings"
            className={`flex items-center ${collapsed ? "justify-center" : "gap-3"} px-3 h-11 rounded-xl transition-all ${
              location.pathname === "/agent/settings"
                ? "bg-[#10B981] text-white shadow-lg shadow-[#10B981]/20"
                : "text-[#6B7280] hover:text-[#E5E7EB] hover:bg-[#1F2937]"
            }`}
            title="Settings"
          >
            <Settings className="w-[18px] h-[18px] flex-shrink-0" />
            {!collapsed && <span className="text-[14px] font-medium">Settings</span>}
          </Link>
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
