// The built-in browser: a source system's headless browser, shown live. It opens on a site
// check (Cloudflare and the like) so you can clear it, or on any page to sign in. Clicks,
// scrolling and typing go to the page; once a check clears, the interrupted action resumes.

import { Dialog } from "@base-ui/react/dialog";
import { ArrowLeft, ArrowRight, Globe, RotateCw, ShieldCheck, X } from "lucide-react";
import { useEffect, useRef, useState, type KeyboardEvent, type MouseEvent, type WheelEvent } from "react";

import { describeError } from "../api/client";
import { sources, type BrowserFrame, type BrowserInput } from "../api/sources";
import { useWorkspace } from "../state/workspace";
import { Button, buttonVariants } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "./ui";

const FRAME_DELAY = 250;
const RESUME_DELAY = 900;
// Keys the page should receive as key presses rather than typed text.
const KEYS = new Set(["Enter", "Backspace", "Delete", "Tab", "Escape", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Home", "End", "PageUp", "PageDown"]);

export function BrowserPanel() {
  const ws = useWorkspace();
  const target = ws.browser!;
  const [session, setSession] = useState<string>();
  const [frame, setFrame] = useState<BrowserFrame>();
  const [error, setError] = useState<string>();
  const [address, setAddress] = useState(target.url);
  const editing = useRef(false);
  const [passed, setPassed] = useState(false);
  const sawCheck = useRef(false);
  const surface = useRef<HTMLDivElement>(null);

  // One session per panel; closing the panel closes it.
  useEffect(() => {
    let alive = true;
    let opened: string | undefined;
    sources.browserOpen(target.system, target.url).then(
      ({ id }) => {
        opened = id;
        if (alive) setSession(id);
        else void sources.browserClose(target.system, id).catch(() => undefined);
      },
      (e) => alive && setError(describeError(e)),
    );
    return () => {
      alive = false;
      if (opened) void sources.browserClose(target.system, opened).catch(() => undefined);
    };
  }, [target.system, target.url]);

  // Frames, one after another so a slow screenshot never piles requests up.
  useEffect(() => {
    if (!session) return;
    let alive = true;
    void (async () => {
      while (alive) {
        try {
          const next = await sources.browserFrame(target.system, session);
          if (!alive) return;
          setFrame(next);
          if (!editing.current) setAddress(next.url);
          if (next.challenge) sawCheck.current = true;
          else if (sawCheck.current && !next.loading && target.resume) {
            setPassed(true);
            window.setTimeout(() => ws.closeBrowser(true), RESUME_DELAY);
            return;
          }
        } catch (e) {
          if (alive) setError(describeError(e));
          return;
        }
        await new Promise((resolve) => window.setTimeout(resolve, FRAME_DELAY));
      }
    })();
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session, target.system]);

  // Input is fire-and-forget: one rejected key or click must not end the session.
  const send = (input: BrowserInput) => {
    if (session) void sources.browserInput(target.system, session, input).catch(() => undefined);
  };

  const point = (event: MouseEvent<HTMLImageElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const scale = (frame?.width ?? rect.width) / rect.width;
    return { x: Math.round((event.clientX - rect.left) * scale), y: Math.round((event.clientY - rect.top) * scale) };
  };

  const onKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (KEYS.has(event.key)) send({ action: "press", key: event.key });
    else if (event.key.length === 1) send({ action: "type", text: event.key });
    else return;
    event.preventDefault();
  };

  const host = (() => {
    try {
      return new URL(frame?.url || target.url).host;
    } catch {
      return target.url;
    }
  })();

  return (
    <Dialog.Root open onOpenChange={(open) => !open && ws.closeBrowser(false)}>
      <Dialog.Portal>
        <Dialog.Backdrop className="fixed inset-0 z-50 bg-[color-mix(in_oklab,var(--n-950)_40%,transparent)] transition-opacity duration-150 ease-settle data-[ending-style]:opacity-0 data-[starting-style]:opacity-0" />
        <Dialog.Popup
          className="fixed top-1/2 left-1/2 z-[51] flex h-[min(920px,calc(100vh_-_48px))] w-[min(1320px,calc(100vw_-_48px))] -translate-x-1/2 -translate-y-1/2 flex-col overflow-hidden rounded-modal bg-surface shadow-[var(--shadow-modal),0_0_0_1px_var(--border-soft)] transition-opacity duration-150 ease-settle data-[ending-style]:opacity-0 data-[starting-style]:opacity-0"
          aria-describedby={undefined}
          initialFocus={surface}
        >
          <div className="flex flex-none items-center gap-1 px-3 py-2 shadow-[inset_0_-1px_0_var(--border-soft)]">
            <Button variant="ghost" size="icon-sm" aria-label="Back" title="Back" disabled={!session} onClick={() => send({ action: "back" })}>
              <ArrowLeft size={16} />
            </Button>
            <Button variant="ghost" size="icon-sm" aria-label="Forward" title="Forward" disabled={!session} onClick={() => send({ action: "forward" })}>
              <ArrowRight size={16} />
            </Button>
            <Button variant="ghost" size="icon-sm" aria-label="Reload" title="Reload" disabled={!session} onClick={() => send({ action: "reload" })}>
              <RotateCw size={16} />
            </Button>
            <form
              className="relative min-w-0 flex-1"
              onSubmit={(event) => {
                event.preventDefault();
                const url = /^https?:\/\//.test(address) ? address : `https://${address}`;
                editing.current = false;
                send({ action: "navigate", url });
              }}
            >
              <Globe size={14} className="pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2 text-icon-2" />
              <Input
                aria-label="Address"
                value={address}
                onFocus={(event) => {
                  editing.current = true;
                  event.target.select();
                }}
                onBlur={() => {
                  editing.current = false;
                  if (frame) setAddress(frame.url);
                }}
                onChange={(event) => setAddress(event.target.value)}
                className="pl-[30px]"
              />
            </form>
            {passed ? (
              <Badge tone="success">Check passed</Badge>
            ) : frame?.challenge ? (
              <Badge tone="warning">Site check</Badge>
            ) : frame?.loading ? (
              <Badge>Loading</Badge>
            ) : null}
            <Dialog.Close className={buttonVariants({ variant: "outline", size: "sm" })}>{target.resume ? "Done" : "Close"}</Dialog.Close>
          </div>
          <Dialog.Title className="sr-only">Browser: {host}</Dialog.Title>
          {frame?.challenge && (
            <p className="flex flex-none items-center gap-1 bg-warning-weak px-3 py-1 text-xs text-warning">
              <ShieldCheck size={14} /> {host} checks that a person is visiting. Complete the check below; Urakatsu continues on its own once it passes.
            </p>
          )}
          <div
            className="flex min-h-0 flex-1 items-center justify-center overflow-hidden bg-canvas focus-visible:shadow-[inset_0_0_0_2px_var(--brand)] focus-visible:outline-none"
            ref={surface}
            tabIndex={0}
            onKeyDown={onKey}
            aria-label={`Page on ${host}`}
          >
            {error ? (
              <div className="flex flex-1 flex-col items-center justify-center gap-1 bg-canvas p-8 text-center">
                <X size={24} />
                <p className="text-base text-fg">The browser stopped.</p>
                <p className="max-w-[42ch] text-fg-2">{error}</p>
              </div>
            ) : frame ? (
              <img
                className="block max-h-full max-w-full cursor-default select-none"
                src={`data:image/jpeg;base64,${frame.image}`}
                alt={frame.title || host}
                draggable={false}
                onClick={(event) => {
                  surface.current?.focus();
                  send({ action: "click", ...point(event) });
                }}
                onWheel={(event: WheelEvent<HTMLImageElement>) => send({ action: "scroll", dx: event.deltaX, dy: event.deltaY })}
              />
            ) : (
              <div className="flex flex-1 flex-col items-center justify-center gap-1 bg-canvas p-8 text-center">
                <div className="relative h-1 w-40 overflow-hidden rounded-full bg-surface-fill" aria-label="Opening the page">
                  <span className="absolute inset-y-0 left-0 w-[30%]! rounded-[inherit] bg-brand transition-[width] duration-200 ease-settle animate-[slide_1.1s_ease-in-out_infinite]" />
                </div>
                <p className="max-w-[42ch] text-fg-2">Opening {host}.</p>
              </div>
            )}
          </div>
        </Dialog.Popup>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
