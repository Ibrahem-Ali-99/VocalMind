import { Link, isRouteErrorResponse, useRouteError } from "react-router";
import { AlertTriangle } from "lucide-react";

function RouteErrorBoundary() {
  const error = useRouteError();

  let title = "Something went wrong";
  let message = "An unexpected error occurred while loading this page.";

  if (isRouteErrorResponse(error)) {
    title = `${error.status} ${error.statusText || "Route Error"}`;
    message = typeof error.data === "string" ? error.data : message;
  } else if (error instanceof Error) {
    message = error.message;
  }

  return (
    <div className="min-h-screen bg-background text-foreground flex items-center justify-center p-6">
      <div className="w-full max-w-xl rounded-2xl border border-border bg-card shadow-sm p-6">
        <div className="flex items-start gap-3">
          <div className="w-10 h-10 rounded-xl bg-destructive/10 text-destructive grid place-items-center">
            <AlertTriangle className="w-5 h-5" />
          </div>
          <div className="min-w-0">
            <h1 className="text-lg font-extrabold">{title}</h1>
            <p className="mt-1 text-sm text-muted-foreground break-words">{message}</p>
          </div>
        </div>
        <div className="mt-5 flex gap-2">
          <Link
            to="/"
            className="inline-flex items-center justify-center px-3 py-2 rounded-lg text-sm font-bold bg-primary text-primary-foreground"
          >
            Go Home
          </Link>
          <Link
            to="/manager"
            className="inline-flex items-center justify-center px-3 py-2 rounded-lg text-sm font-bold border border-border text-foreground"
          >
            Open Dashboard
          </Link>
        </div>
      </div>
    </div>
  );
}

export { RouteErrorBoundary };
export default RouteErrorBoundary;
