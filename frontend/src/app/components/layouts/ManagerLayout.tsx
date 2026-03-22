import { Outlet, Link, useLocation } from "react-router";
import { useState } from "react";
import { useAuth } from "../../contexts/AuthContext";
import { UserNav } from "./UserNav";
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
  const { user } = useAuth();
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();

  const getInitials = (name: string) => {
    if (!name) return "??";
    const parts = name.split(" ");
    if (parts.length >= 2) return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
    return name.substring(0, 2).toUpperCase();
  };

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
    <div className="flex h-screen bg-background text-foreground transition-colors duration-300">
      {/* Sidebar */}
      <div
        className={`${
          collapsed ? "w-[72px]" : "w-[240px]"
        } bg-sidebar border-r border-sidebar-border flex flex-col transition-all duration-300`}
      >
        {/* Logo Area */}
        <div className="h-16 flex items-center px-4 border-b border-sidebar-border">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-primary rounded-xl flex items-center justify-center flex-shrink-0">
              <Mic className="w-[18px] h-[18px] text-primary-foreground" />
            </div>
            {!collapsed && (
              <span className="text-sidebar-foreground font-bold text-lg" style={{ fontFamily: 'var(--font-sans)' }}>
                VocalMind
              </span>
            )}
          </div>
        </div>

        {/* Navigation & Content Area (Flex Grow) */}
        <div className="flex-1 flex flex-col justify-between py-4">
          {/* Top Group: Role Badge & Nav */}
          <div className="space-y-6">
            {!collapsed && (
              <div className="px-4">
                <div className="bg-primary/10 border border-primary/20 rounded-xl p-3.5 transition-all shadow-sm">
                  <div className="text-[10px] font-bold text-primary uppercase tracking-widest mb-1 opacity-80">
                    Manager Portal
                  </div>
                  <div className="text-[12px] text-foreground font-semibold leading-tight">
                    Full org access
                  </div>
                </div>
              </div>
            )}

            <nav className="px-2 space-y-1.5">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.path || 
                  (item.path === "/manager/inspector" && location.pathname.includes("/manager/inspector"));
                
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`flex items-center gap-3 px-3.5 h-11 rounded-xl transition-all ${
                      isActive
                        ? "bg-primary text-primary-foreground shadow-lg shadow-primary/20"
                        : "text-sidebar-foreground/60 hover:text-sidebar-foreground hover:bg-sidebar-accent"
                    }`}
                  >
                    <Icon className="w-[18px] h-[18px] flex-shrink-0" />
                    {!collapsed && (
                      <span className="text-[14px] font-semibold">
                        {item.label}
                      </span>
                    )}
                  </Link>
                );
              })}
            </nav>
          </div>

          {/* Bottom Group (Distributed) */}
          <div className="px-2 space-y-4">
            {/* Additional items could go here if needed to fill space */}
          </div>
        </div>

        {/* Bottom Section */}
        <div className="border-t border-sidebar-border p-4">
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="w-full flex items-center justify-center h-8 text-sidebar-foreground/60 hover:text-sidebar-foreground transition-colors mb-3"
          >
            {collapsed ? (
              <ChevronRight className="w-5 h-5" />
            ) : (
              <ChevronLeft className="w-5 h-5" />
            )}
          </button>

          {!collapsed && (
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-primary-foreground text-[11px] font-bold ring-2 ring-white/10">
                {getInitials(user?.name || "Manager")}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sidebar-foreground text-[13px] font-semibold truncate leading-tight">
                  {user?.name || "Manager King"}
                </div>
                <div className="text-sidebar-foreground/60 text-[11px] font-medium capitalize mt-0.5">
                  {user?.role || "Manager"}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Bar */}
        <div className="h-14 bg-card border-b border-border px-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-[16px] font-bold text-foreground">
              {getPageTitle()}
            </h1>
            <span className="px-2.5 py-1 bg-primary/10 text-primary border border-primary/20 rounded-full text-[11px] font-semibold uppercase tracking-wide">
              Manager
            </span>
          </div>

          <div className="flex items-center gap-3">
            <button className="flex items-center gap-2 px-3 h-8 border border-border rounded-lg text-[13px] text-muted-foreground hover:bg-accent transition-colors">
              <Download className="w-3.5 h-3.5" />
              Export
            </button>
            <button className="w-8 h-8 flex items-center justify-center bg-accent/30 border border-border rounded-lg hover:bg-accent transition-colors">
              <Bell className="w-4 h-4 text-muted-foreground" />
            </button>
            <UserNav />
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
