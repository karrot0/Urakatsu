// Settings > Sources > Komga: the Komga server projects are exported to (urakatsu/sources/komga).
// Komga reads books from its libraries' folders, so besides the address and an API key, each library
// needs its folder as this computer sees it: found by itself when Komga runs here, chosen once when
// Komga runs in Docker or on a NAS.

import { Eye, EyeOff, RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";

import { describeError } from "../api/client";
import { komga, type KomgaLibrary, type KomgaStatus } from "../api/komga";
import { useWorkspace } from "../state/workspace";
import { PreferenceRow, PreferenceSection } from "./SettingsPage";
import { Badge } from "./ui";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

/** The Komga sections of the Sources page: the server, then its libraries once connected. */
export function KomgaSettings() {
  const ws = useWorkspace();
  const [status, setStatus] = useState<KomgaStatus>();
  const [busy, setBusy] = useState(false);

  const act = async (task: () => Promise<KomgaStatus>) => {
    setBusy(true);
    try {
      setStatus(await task());
      return true;
    } catch (error) {
      ws.reportError(describeError(error));
      return false;
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    void act(komga.status);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const badge = !status ? null : status.connected ? (
    <Badge tone="success">Connected</Badge>
  ) : status.configured ? (
    <Badge tone="danger">Not connected</Badge>
  ) : (
    <Badge tone="neutral">Not set up</Badge>
  );

  return (
    <>
      <PreferenceSection
        title="Komga"
        description="Export projects to a Komga server as series, one book per chapter; automations can export the chapters they finish. Use an API key from an admin account (in Komga: Account Settings, API Keys); it is kept on this computer and never shown again."
        action={badge}
      >
        {!status && <p className="py-3 text-fg-2">Loading Komga settings.</p>}
        {status && (
          <>
            <ConnectionForm status={status} busy={busy} onSave={(url, key) => act(() => komga.connect(url, key))} />
            {status.connected && (
              <PreferenceRow title="Signed in as" description={status.version ? `Komga ${status.version}` : undefined}>
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate">{status.user}</span>
                  <Button variant="ghost" size="sm" disabled={busy} onClick={() => void act(komga.status)}>
                    <RefreshCw size={14} /> Check again
                  </Button>
                </div>
              </PreferenceRow>
            )}
            {status.error && (
              <PreferenceRow title="Problem">
                <span className="text-xs text-danger">{status.error}</span>
              </PreferenceRow>
            )}
          </>
        )}
      </PreferenceSection>
      {status?.connected && (
        <PreferenceSection
          title="Komga libraries"
          description="Komga only reads books from its libraries' folders, so Urakatsu writes them there. When Komga runs in Docker or on another computer, choose where each library's folder is on this one."
        >
          {status.libraries.length === 0 && <p className="py-3 text-fg-2">This Komga has no libraries yet. Add one in Komga, then check again.</p>}
          {status.libraries.map((library) => (
            <LibraryRow key={library.id} library={library} busy={busy} onFolder={(folder) => act(() => komga.setFolder(library.id, folder))} />
          ))}
        </PreferenceSection>
      )}
    </>
  );
}

function ConnectionForm({ status, busy, onSave }: { status: KomgaStatus; busy: boolean; onSave: (url: string, key: string | null) => Promise<boolean> }) {
  const [url, setUrl] = useState(status.url);
  const [key, setKey] = useState("");
  const [shown, setShown] = useState(false);
  useEffect(() => setUrl(status.url), [status.url]);
  const moved = url.trim().replace(/\/+$/, "") !== status.url;
  const changed = moved || key.trim() !== "";
  // The saved key is only ever sent to the address it was saved for: a new address needs its own.
  const keyNeeded = (!status.configured || moved) && !key.trim();
  return (
    <form
      onSubmit={async (event) => {
        event.preventDefault();
        if (await onSave(url.trim(), key.trim() || null)) setKey("");
      }}
    >
      <PreferenceRow title="Address" description="Such as http://localhost:25600.">
        <Input aria-label="Komga address" placeholder="http://localhost:25600" value={url} onChange={(e) => setUrl(e.target.value)} />
      </PreferenceRow>
      <PreferenceRow
        title="API key"
        description={status.configured ? (moved ? "Enter the key for the new address; the saved key stays with the old one." : "Saved. Enter a new one to replace it.") : undefined}
      >
        <div className="flex gap-0.5">
          <Input
            aria-label="Komga API key"
            type={shown ? "text" : "password"}
            autoComplete="off"
            placeholder={status.configured && !moved ? "Saved" : "Not set"}
            value={key}
            onChange={(e) => setKey(e.target.value)}
          />
          <Button type="button" variant="ghost" size="icon-sm" aria-label={shown ? "Hide the API key" : "Show the API key"} onClick={() => setShown((s) => !s)}>
            {shown ? <EyeOff size={14} /> : <Eye size={14} />}
          </Button>
        </div>
      </PreferenceRow>
      <div className="flex justify-end py-3">
        <Button type="submit" size="sm" disabled={busy || !url.trim() || (!changed && status.connected) || keyNeeded}>
          {status.configured ? "Save and connect" : "Connect"}
        </Button>
      </div>
    </form>
  );
}

function LibraryRow({ library, busy, onFolder }: { library: KomgaLibrary; busy: boolean; onFolder: (folder: string | null) => Promise<boolean> }) {
  const [folder, setFolder] = useState(library.folder_source === "set" ? (library.folder ?? "") : "");
  useEffect(() => setFolder(library.folder_source === "set" ? (library.folder ?? "") : ""), [library.folder, library.folder_source]);
  const description = library.problem
    ? undefined
    : library.folder_source === "komga"
      ? `Komga runs on this computer: books go to ${library.folder}.`
      : `Books go to ${library.folder}.`;
  return (
    <PreferenceRow title={library.name} description={`Komga reads ${library.root}`} align="start">
      <form
        className="flex flex-col gap-1.5"
        onSubmit={(event) => {
          event.preventDefault();
          void onFolder(folder.trim() || null);
        }}
      >
        <div className="flex gap-1">
          <Input
            className="min-w-0 flex-1"
            aria-label={`Folder of ${library.name} on this computer`}
            placeholder={library.folder_source === "komga" ? (library.folder ?? "") : "Folder on this computer"}
            value={folder}
            onChange={(e) => setFolder(e.target.value)}
          />
          <Button type="submit" variant="outline" size="sm" disabled={busy || folder.trim() === (library.folder_source === "set" ? library.folder : "")}>
            Save
          </Button>
        </div>
        {library.problem ? <span className="text-xs text-[var(--warning)]">{library.problem}</span> : <span className="text-xs text-fg-3">{description}</span>}
      </form>
    </PreferenceRow>
  );
}
