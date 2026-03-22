import { createBrowserRouter } from "react-router";
import { ManagerLayout } from "./components/layouts/ManagerLayout";
import { AgentLayout } from "./components/layouts/AgentLayout";
import { ManagerDashboard } from "./components/manager/ManagerDashboard";
import { SessionInspector } from "./components/manager/SessionInspector";
import { SessionDetail } from "./components/manager/SessionDetail";
import { ManagerAssistant } from "./components/manager/ManagerAssistant";
import { KnowledgeBase } from "./components/manager/KnowledgeBase";
import { AgentDashboard } from "./components/agent/AgentDashboard";
import { AgentCallDetail } from "./components/agent/AgentCallDetail";
import { LandingPage } from "./components/LandingPage";

import { ProtectedRoute } from "./components/ProtectedRoute";
import { AuthProvider } from "./contexts/AuthContext";
import { SettingsPage } from "./components/SettingsPage";
import { UnderDevelopment } from "./components/ui/UnderDevelopment";
import Login from "./pages/Login";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <LandingPage />,
  },
  {
    path: "/login",
    element: <Login />,
  },
  {
    element: <ProtectedRoute />,
    children: [
      {
        path: "/manager",
        element: <ManagerLayout />,
        children: [
          { index: true, element: <ManagerDashboard /> },
          { path: "inspector", element: <SessionInspector /> },
          { path: "inspector/:id", element: <SessionDetail /> },
          { path: "assistant", element: <ManagerAssistant /> },
          { path: "knowledge", element: <KnowledgeBase /> },
          { path: "settings", element: <SettingsPage /> },
          { path: "*", element: <UnderDevelopment /> },
        ],
      },
      {
        path: "/agent",
        element: <AgentLayout />,
        children: [
          { index: true, element: <AgentDashboard /> },
          { path: "calls", element: <UnderDevelopment /> },
          { path: "calls/:id", element: <AgentCallDetail /> },
          { path: "settings", element: <SettingsPage /> },
          { path: "*", element: <UnderDevelopment /> },
        ],
      },
    ],
  },
  {
    path: "*",
    element: <UnderDevelopment />,
  },
]);
