import React from "react";
import { Link } from "react-router";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "../ui/dropdown-menu";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "../ui/alert-dialog";
import { useAuth } from "../../contexts/AuthContext";
import { useTheme } from "../../contexts/ThemeContext";
import { 
  LogOut, 
  User as UserIcon, 
  Settings, 
  Moon, 
  Sun, 
  Monitor,
  Check
} from "lucide-react";

export function UserNav() {
  const { user, logout } = useAuth();
  const { theme, setTheme } = useTheme();

  const getInitials = (name: string) => {
    if (!name) return "??";
    const parts = name.split(" ");
    if (parts.length >= 2) return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
    return name.substring(0, 2).toUpperCase();
  };

  return (
    <div className="flex items-center gap-2">
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button className="relative flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground text-[11px] font-bold ring-2 ring-primary/20 hover:ring-primary/40 transition-all focus:outline-none">
            {getInitials(user?.name || "User")}
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent className="w-56 bg-popover/95 backdrop-blur-xl border-border" align="end" forceMount>
          <DropdownMenuLabel className="font-normal">
            <div className="flex flex-col space-y-1">
              <p className="text-sm font-semibold text-foreground leading-none">
                {user?.name || "User Name"}
              </p>
              <p className="text-xs text-muted-foreground leading-none truncate">
                {user?.email}
              </p>
            </div>
          </DropdownMenuLabel>
          <DropdownMenuSeparator className="bg-border" />
          <DropdownMenuGroup>
            <DropdownMenuItem asChild className="text-foreground hover:bg-accent transition-colors cursor-pointer">
              <Link to={`${user?.role === "manager" ? "/manager/settings" : "/agent/settings"}#profile`}>
                <UserIcon className="mr-2 h-4 w-4" />
                <span>Profile</span>
              </Link>
            </DropdownMenuItem>
            <DropdownMenuItem asChild className="text-foreground hover:bg-accent transition-colors cursor-pointer">
              <Link to={user?.role === "manager" ? "/manager/settings" : "/agent/settings"}>
                <Settings className="mr-2 h-4 w-4" />
                <span>Settings</span>
              </Link>
            </DropdownMenuItem>
          </DropdownMenuGroup>
          <DropdownMenuSeparator className="bg-border" />
          
          <DropdownMenuLabel className="text-[11px] uppercase tracking-wider text-muted-foreground py-1.5 font-bold">
            Appearance
          </DropdownMenuLabel>
          <DropdownMenuItem 
            onClick={() => setTheme("light")}
            className="flex items-center justify-between text-foreground hover:bg-accent transition-colors cursor-pointer"
          >
            <div className="flex items-center">
              <Sun className="mr-2 h-4 w-4" />
              <span>Light</span>
            </div>
            {theme === "light" && <Check className="h-4 w-4 text-primary" />}
          </DropdownMenuItem>
          <DropdownMenuItem 
            onClick={() => setTheme("dark")}
            className="flex items-center justify-between text-foreground hover:bg-accent transition-colors cursor-pointer"
          >
            <div className="flex items-center">
              <Moon className="mr-2 h-4 w-4" />
              <span>Dark</span>
            </div>
            {theme === "dark" && <Check className="h-4 w-4 text-primary" />}
          </DropdownMenuItem>
          <DropdownMenuItem 
            onClick={() => setTheme("system")}
            className="flex items-center justify-between text-foreground hover:bg-accent transition-colors cursor-pointer"
          >
            <div className="flex items-center">
              <Monitor className="mr-2 h-4 w-4" />
              <span>System</span>
            </div>
            {theme === "system" && <Check className="h-4 w-4 text-primary" />}
          </DropdownMenuItem>
          
          <DropdownMenuSeparator className="bg-border" />
          
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <DropdownMenuItem 
                onSelect={(e) => e.preventDefault()}
                className="text-destructive hover:bg-destructive/10 transition-colors cursor-pointer font-medium"
              >
                <LogOut className="mr-2 h-4 w-4" />
                <span>Log out</span>
              </DropdownMenuItem>
            </AlertDialogTrigger>
            <AlertDialogContent className="bg-popover/95 backdrop-blur-xl border-border">
              <AlertDialogHeader>
                <AlertDialogTitle className="text-foreground">Log Out</AlertDialogTitle>
                <AlertDialogDescription className="text-muted-foreground">
                  Are you sure you want to log out of your session?
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel className="border-border text-foreground hover:bg-accent">Cancel</AlertDialogCancel>
                <AlertDialogAction 
                  onClick={logout}
                  className="bg-destructive hover:opacity-90 text-destructive-foreground border-none"
                >
                  Sign Out
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
