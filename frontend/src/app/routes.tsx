import { createBrowserRouter } from "react-router";
import { ManagerLayout } from "./components/layouts/ManagerLayout";
import { AgentLayout } from "./components/layouts/AgentLayout";
import { ManagerDashboard } from "./components/manager/ManagerDashboard";
import { SessionInspector } from "./components/manager/SessionInspector";
import { SessionDetail } from "./components/manager/SessionDetail";
import { ManagerAssistant } from "./components/manager/ManagerAssistant";
import { KnowledgeBase } from "./components/manager/KnowledgeBase";
import { ManagerSettings } from "./components/manager/ManagerSettings";
import { AgentDashboard } from "./components/agent/AgentDashboard";
import { AgentCallDetail } from "./components/agent/AgentCallDetail";
import { LandingPage } from "./components/LandingPage";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <LandingPage />,
  },
  {
    path: "/manager",
    element: <ManagerLayout />,
    children: [
      { index: true, element: <ManagerDashboard /> },
      { path: "inspector", element: <SessionInspector /> },
      { path: "inspector/:id", element: <SessionDetail /> },
      { path: "assistant", element: <ManagerAssistant /> },
      { path: "knowledge", element: <KnowledgeBase /> },
      { path: "settings", element: <ManagerSettings /> },
    ],
  },
  {
    path: "/agent",
    element: <AgentLayout />,
    children: [
      { index: true, element: <AgentDashboard /> },
      { path: "calls/:id", element: <AgentCallDetail /> },
    ],
  },
]);
