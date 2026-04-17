import { Fragment, useState } from "react";
import { Outlet, Link, useLocation } from "react-router";
import { UserNav } from "./UserNav";
import {
  LayoutDashboard,
  Search,
  MessageSquare,
  BookOpen,
  ChevronLeft,
  ChevronRight,
  Bell,
  Download,
} from "lucide-react";
import logoSrc from "../../../assets/logo/logo.svg";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "../ui/dropdown-menu";
import { Separator } from "../ui/separator";

export function ManagerLayout() {
  const [collapsed, setCollapsed] = useState(false);
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
    <div className="flex h-screen bg-background text-foreground transition-colors duration-300">
      {/* Sidebar */}
      <div
        className={`${
          collapsed ? "w-[72px]" : "w-[240px]"
        } bg-sidebar border-r border-sidebar-border flex flex-col transition-all duration-300`}
      >
        {/* Logo Area */}
        <div className="h-16 flex items-center px-4 border-b border-sidebar-border">
          <Link to="/manager" className="flex items-center gap-3 w-full">
            <img src={logoSrc} alt="VocalMind" className="w-[32px] h-[32px] object-contain flex-shrink-0" />
            {!collapsed && (
              <span className="text-sidebar-foreground font-bold text-lg" style={{ fontFamily: 'var(--font-sans)' }}>
                VocalMind
              </span>
            )}
          </Link>
        </div>

        {/* Navigation & Content Area (Flex Grow) */}
        <div className="flex-1 flex flex-col justify-between py-4">
          <div>
            <nav className="px-2 flex flex-col gap-0.5" aria-label="Manager navigation">
              {navItems.map((item, index) => {
                const Icon = item.icon;
                const isActive =
                  location.pathname === item.path ||
                  (item.path === "/manager/inspector" &&
                    location.pathname.includes("/manager/inspector"));

                return (
                  <Fragment key={item.path}>
                    {index > 0 && (
                      <Separator className="my-2 bg-sidebar-border/60" decorative />
                    )}
                    <Link
                      to={item.path}
                      className={`flex items-center gap-3 px-3.5 h-11 rounded-xl transition-colors ${
                        isActive
                          ? "bg-primary text-primary-foreground shadow-lg shadow-primary/20"
                          : "text-sidebar-foreground/60 hover:text-sidebar-foreground hover:bg-sidebar-accent"
                      }`}
                    >
                      <Icon className="w-[18px] h-[18px] flex-shrink-0" />
                      {!collapsed && (
                        <span className="text-[14px] font-semibold truncate">{item.label}</span>
                      )}
                    </Link>
                  </Fragment>
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
            data-cy="sidebar-collapse-toggle"
            className="w-full flex items-center justify-center h-8 text-sidebar-foreground/60 hover:text-sidebar-foreground transition-colors mb-3"
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? (
              <ChevronRight className="w-5 h-5" />
            ) : (
              <ChevronLeft className="w-5 h-5" />
            )}
          </button>

        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        {/* Top Bar */}
        <div className="h-16 bg-card border-b border-border px-6 flex items-center justify-between">
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
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button className="w-8 h-8 flex items-center justify-center bg-accent/30 border border-border rounded-lg hover:bg-accent transition-colors">
                  <Bell className="w-4 h-4 text-muted-foreground" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-64 bg-card border-border shadow-lg" align="end">
                <DropdownMenuLabel>Notifications</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <div className="p-4 text-center text-sm text-muted-foreground">
                  No new notifications
                </div>
              </DropdownMenuContent>
            </DropdownMenu>
            <UserNav />
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto min-w-0">
          <Outlet />
        </div>
      </div>
    </div>
  );
}
