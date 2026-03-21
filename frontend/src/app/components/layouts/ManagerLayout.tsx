import { Outlet, Link, useLocation } from "react-router";
import { useState } from "react";
import {
  Menu,
  LayoutDashboard,
  Search,
  MessageSquare,
  BookOpen,
  Settings,
  Bell,
  Download,
  User,
  LogOut,
  ChevronDown
} from "lucide-react";
import logoSrc from "../../../assets/logo/logo.svg";

export function ManagerLayout() {
  const [collapsed, setCollapsed] = useState(false);
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isNotifOpen, setIsNotifOpen] = useState(false);
  const location = useLocation();

  const navItems = [
    { icon: LayoutDashboard, label: "Dashboard", path: "/manager" },
    { icon: Search, label: "Session Inspector", path: "/manager/inspector" },
    { icon: MessageSquare, label: "Manager Assistant", path: "/manager/assistant" },
    { icon: BookOpen, label: "Knowledge Base", path: "/manager/knowledge" },
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
          collapsed ? "w-[80px]" : "w-[248px]"
        } bg-[#0D1117] border-r border-[#1F2937] flex flex-col transition-all duration-300`}
      >
        {/* Logo + Menu Header */}
        <div className={`border-b border-[#1F2937] ${collapsed ? "px-2 py-4" : "px-4 py-4"}`}>
          <Link
            to="/manager"
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
        <nav className="flex-1 px-2 py-3 space-y-1.5">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path || 
              (item.path === "/manager/inspector" && location.pathname.includes("/manager/inspector"));
            
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center ${collapsed ? "justify-center" : "gap-3"} px-3 h-11 rounded-xl transition-all ${
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

        {/* Bottom Settings */}
        <div className="border-t border-[#1F2937] p-2">
          <Link
            to="/manager/settings"
            className={`flex items-center ${collapsed ? "justify-center" : "gap-3"} px-3 h-11 rounded-xl transition-all ${
              location.pathname === "/manager/settings"
                ? "bg-[#3B82F6] text-white shadow-lg shadow-[#3B82F6]/20"
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

            {/* Notification Dropdown */}
            <div className="relative">
              <button 
                onClick={() => { setIsNotifOpen(!isNotifOpen); setIsProfileOpen(false); }}
                className="w-8 h-8 flex items-center justify-center bg-[#F9FAFB] border border-[#E5E7EB] rounded-lg hover:bg-[#F3F4F6] transition-colors relative"
              >
                <Bell className="w-4 h-4 text-[#6B7280]" />
                <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 bg-red-500 rounded-full border border-white"></span>
              </button>
              {isNotifOpen && (
                <div className="absolute right-0 mt-2 w-72 bg-white border border-[#E5E7EB] rounded-xl shadow-lg z-50 py-2">
                  <div className="px-4 py-2 border-b border-[#E5E7EB]">
                    <h3 className="text-sm font-semibold text-[#111827]">Notifications</h3>
                  </div>
                  <div className="max-h-64 overflow-y-auto">
                    <div className="px-4 py-3 hover:bg-[#F9FAFB] transition-colors cursor-pointer">
                      <p className="text-xs text-[#374151]">System updated default SOP guidelines.</p>
                      <p className="text-[10px] text-[#9CA3AF] mt-1">2 mins ago</p>
                    </div>
                    <div className="px-4 py-3 hover:bg-[#F9FAFB] transition-colors cursor-pointer">
                      <p className="text-xs text-[#374151]">Agent Dina completed 5 sessions.</p>
                      <p className="text-[10px] text-[#9CA3AF] mt-1">1 hour ago</p>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Profile Dropdown */}
            <div className="relative">
              <button 
                onClick={() => { setIsProfileOpen(!isProfileOpen); setIsNotifOpen(false); }}
                className="flex items-center gap-2 hover:bg-[#F3F4F6] p-1 rounded-lg transition-colors cursor-pointer"
              >
                <div className="w-8 h-8 rounded-full bg-[#3B82F6] flex items-center justify-center text-white text-xs font-semibold">
                  MK
                </div>
                <ChevronDown className="w-4 h-4 text-[#6B7280]" />
              </button>

              {isProfileOpen && (
                <div className="absolute right-0 mt-2 w-48 bg-white border border-[#E5E7EB] rounded-xl shadow-lg z-50 py-1">
                  <div className="px-4 py-3 border-b border-[#E5E7EB]">
                    <p className="text-sm font-medium text-[#111827]">Manager User</p>
                    <p className="text-xs text-[#6B7280] truncate">manager@vocalmind.io</p>
                  </div>
                  <Link 
                    to="/manager/settings" 
                    className="flex items-center gap-2 px-4 py-2 text-sm text-[#374151] hover:bg-[#F3F4F6] transition-colors"
                    onClick={() => setIsProfileOpen(false)}
                  >
                    <User className="w-4 h-4" /> Edit Profile
                  </Link>
                  <Link 
                    to="/manager/settings" 
                    className="flex items-center gap-2 px-4 py-2 text-sm text-[#374151] hover:bg-[#F3F4F6] transition-colors"
                    onClick={() => setIsProfileOpen(false)}
                  >
                    <Settings className="w-4 h-4" /> Settings
                  </Link>
                  <div className="border-t border-[#E5E7EB] my-1"></div>
                  <button className="w-full flex items-center gap-2 px-4 py-2 text-sm text-[#EF4444] hover:bg-[#FEF2F2] transition-colors">
                    <LogOut className="w-4 h-4" /> Log out
                  </button>
                </div>
              )}
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
