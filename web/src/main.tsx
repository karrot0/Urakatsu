import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "./desktop";
import { App } from "./App";
import { WorkspaceProvider } from "./state/workspace";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <WorkspaceProvider>
      <App />
    </WorkspaceProvider>
  </StrictMode>,
);
