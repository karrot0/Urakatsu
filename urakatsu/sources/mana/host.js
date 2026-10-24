// Runs Mana source bundles for Urakatsu under Bun.
//
// A built `.mana` bundle is self-contained JavaScript ending in
// `globalThis.Target = __exports__.Target`. Its HttpClient talks to the host through
// `globalThis.__manaHTTPBridge`, and the source expects `ObjectStore`, `SecureStore` and
// `WebViewPage` as globals. Those shims mirror the contract the Mana app provides (and the
// kakarot-extensions-mana verify harness).
//
// The built-in browser. The Mana app answers site checks (Cloudflare and the like) with a
// WebView; here that is a headless Chromium driven through Bun.WebView. When a request comes
// back as a challenge, the host opens the site in a browser tab, waits for the check to clear,
// and from then on sends that site's requests from inside the tab: a cleared check is bound to
// the browser's own network stack, so its cookies alone do not get a plain fetch through. A
// check that needs a person is reported as a challenge; Urakatsu shows the page in an
// interactive session (browser.* methods) and the user clears it there.
//
// Protocol: one JSON object per line. Requests on stdin are {id, method, params}; replies
// on stdout are {id, result} or {id, error, challenge?, needsBrowser?}. Logs go to stderr.

import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

const METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"];
const Operation = { Fetch: 0, Read: 1, Release: 2, Cancel: 3, CreateBody: 4 };
const BODY_BYTES = 3;
const READER_BASE64 = 1;
const ORIGIN_COVER = 1;
const ORIGIN_PAGE = 2;
const IMAGE_DISPATCH = Symbol.for("mana.http.image.v1");
const VIEWPORT = { width: 1280, height: 800 };
const AUTO_CLEAR_MS = 25_000; // how long an automatic check may take before a person is asked
const BROWSER_IDLE_MS = 10 * 60_000;
const MAX_TABS = 6;
const CHALLENGE_TITLE = /^\s*(just a moment|attention required|checking your browser)/i;
const CHALLENGE_URL = /\/cdn-cgi\/challenge-platform|[?&]__cf_chl|\/_c\?t=/;
const CHALLENGE_BODY = /<title>\s*(just a moment|attention required)|challenge-platform|cf_chl_opt|window\._cf_chl/i;

let userAgent =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36";

const log = (...args) => console.error("[mana-host]", ...args);

class ChallengeError extends Error {
  constructor(url) {
    super(`${new URL(url).host} asks for a browser check that needs you`);
    this.name = "ChallengeError";
    this.url = url;
  }
}

class BrowserUnavailable extends Error {
  constructor() {
    super("this site needs a browser and none is set up");
    this.name = "BrowserUnavailable";
  }
}

function decodeStrictUtf8(buffer) {
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(buffer);
  } catch {
    throw new Error("String could not be serialized with encoding: Unicode (UTF-8)");
  }
}

function challenged(status, url, headers, bytes) {
  if (headers["cf-mitigated"] === "challenge" || CHALLENGE_URL.test(url)) return true;
  if (![403, 429, 503].includes(status)) return false;
  return CHALLENGE_BODY.test(new TextDecoder().decode(bytes.subarray(0, 16384)));
}

// ---- browser ------------------------------------------------------------------------------------

const browser = {
  path: process.env.URAKATSU_BROWSER_PATH || null,
  profile: process.env.URAKATSU_BROWSER_PROFILE || null,
  name: null,
  version: null,
  lastUse: 0,
};
const tabs = new Map(); // origin -> {view, cleared, lock}
const routed = new Set(); // origins whose requests run inside a browser tab
const sessions = new Map(); // interactive sessions shown in Urakatsu
let nextSession = 0;

const VIEW_METHODS = ["navigate", "evaluate", "screenshot", "click", "type", "press", "scroll", "goBack", "goForward", "reload", "cdp"];

/** Bun.WebView runs one operation at a time per view and rejects overlaps; queue them instead. */
function serialized(view) {
  let chain = Promise.resolve();
  const run = (task) => {
    const next = chain.then(task, task);
    chain = next.catch(() => undefined);
    return next;
  };
  const wrapped = { close: () => view.close() };
  for (const name of VIEW_METHODS) wrapped[name] = (...args) => run(() => view[name](...args));
  return wrapped;
}

async function newView() {
  if (!browser.path) throw new BrowserUnavailable();
  browser.lastUse = Date.now();
  const raw = new Bun.WebView({
    ...VIEWPORT,
    backend: { type: "chrome", path: browser.path, argv: ["--disable-blink-features=AutomationControlled", "--no-first-run"] },
    dataStore: { directory: browser.profile },
  });
  const view = serialized(raw);
  await view.navigate("about:blank");
  if (!browser.version) {
    const version = await view.cdp("Browser.getVersion");
    browser.version = version.product;
    // A headless browser names itself in its user agent; sites check for that.
    userAgent = version.userAgent.replace("HeadlessChrome", "Chrome");
  }
  await view.cdp("Emulation.setUserAgentOverride", { userAgent });
  await view.cdp("Network.enable");
  return view;
}

async function pageState(view) {
  try {
    return await view.evaluate("({ title: document.title, url: location.href, ready: document.readyState })");
  } catch {
    return { title: "", url: "", ready: "loading" };
  }
}

const blockedPage = (state) => CHALLENGE_TITLE.test(state.title) || CHALLENGE_URL.test(state.url);

/** Waits for a page to get past any check on its own; false when it still shows one. */
async function settle(view, timeout = AUTO_CLEAR_MS) {
  const deadline = Date.now() + timeout;
  await Bun.sleep(500);
  while (Date.now() < deadline) {
    const state = await pageState(view);
    if (state.ready === "complete" && state.url && !state.url.startsWith("about:") && !blockedPage(state)) return true;
    await Bun.sleep(750);
  }
  return false;
}

function withLock(holder, task) {
  const run = holder.lock.then(task, task);
  holder.lock = run.catch(() => undefined);
  return run;
}

function closeTab(origin) {
  const tab = tabs.get(origin);
  tabs.delete(origin);
  try {
    tab?.view?.close();
  } catch {
    /* already gone */
  }
}

/** A tab on ``origin`` that has cleared its check, opened at ``url`` when it needs (re)loading. */
async function clearedTab(origin, url, referer) {
  let tab = tabs.get(origin);
  if (!tab) {
    if (tabs.size >= MAX_TABS) closeTab(tabs.keys().next().value);
    tab = { view: null, cleared: false, lock: Promise.resolve() };
    tabs.set(origin, tab);
  }
  return withLock(tab, async () => {
    tab.view ??= await newView();
    if (!tab.cleared) {
      if (referer) await tab.view.cdp("Page.navigate", { url, referrer: referer });
      else await tab.view.navigate(url).catch((error) => log("navigate", url, error?.message));
      tab.cleared = (await settle(tab.view)) && new URL((await pageState(tab.view)).url).origin === origin;
      if (!tab.cleared) throw new ChallengeError(url);
    }
    return tab;
  });
}

// Runs inside the site's own tab: same origin, so the browser sends its cookies itself.
function pageFetch(request) {
  // Headers a page's fetch may not set; the browser supplies its own.
  const forbidden = /^(user-agent|cookie|host|origin|referer|connection|content-length|accept-encoding|keep-alive|te|trailer|upgrade|via|sec-.*|proxy-.*)$/i;
  return (async () => {
    const headers = new Headers();
    for (const [name, value] of Object.entries(request.headers ?? {})) {
      if (!forbidden.test(name)) headers.set(name, value);
    }
    let body;
    if (request.body !== undefined && request.body !== null) {
      body = request.bytes ? Uint8Array.from(atob(request.body), (c) => c.charCodeAt(0)) : request.body;
    }
    const response = await fetch(request.url, {
      method: request.method,
      headers,
      body,
      credentials: "include",
      redirect: "follow",
    });
    const buffer = new Uint8Array(await response.arrayBuffer());
    let binary = "";
    for (let i = 0; i < buffer.length; i += 0x8000) binary += String.fromCharCode.apply(null, buffer.subarray(i, i + 0x8000));
    return { status: response.status, url: response.url, headers: Object.fromEntries(response.headers.entries()), body: btoa(binary) };
  })();
}

async function browserFetch(request) {
  const origin = new URL(request.url).origin;
  const start = request.method === "GET" ? request.url : `${origin}/`;
  for (let attempt = 0; attempt < 2; attempt++) {
    const referer = Object.entries(request.headers ?? {}).find(([name]) => /^referer$/i.test(name))?.[1];
    const tab = await clearedTab(origin, start, referer);
    const script = `(${pageFetch.toString()})(${JSON.stringify(request)})`;
    browser.lastUse = Date.now();
    // A page's fetch cannot send another site's Referer (image hosts check it), so the tab sets it.
    const result = referer
      ? await withLock(tab, async () => {
          await tab.view.cdp("Network.setExtraHTTPHeaders", { headers: { Referer: referer } });
          try {
            return await tab.view.evaluate(script);
          } finally {
            await tab.view.cdp("Network.setExtraHTTPHeaders", { headers: {} });
          }
        })
      : await tab.view.evaluate(script);
    const bytes = new Uint8Array(Buffer.from(result.body, "base64"));
    if (result.status >= 400) log("browser", request.method, request.url, "->", result.status, new TextDecoder().decode(bytes.subarray(0, 200)));
    if (!challenged(result.status, result.url, result.headers, bytes)) return { ...result, bytes };
    tab.cleared = false; // the clearance expired; load the site again once
  }
  throw new ChallengeError(request.url);
}

/** Every request a source makes: plain fetch, or the browser for sites that check. */
async function transportFetch(request, state, signal) {
  const origin = new URL(request.url).origin;
  if (routed.has(origin) && browser.path) return browserFetch(request);
  const body = request.body === undefined || request.body === null ? undefined : request.bytes ? Buffer.from(request.body, "base64") : request.body;
  const raw = await fetch(request.url, {
    method: request.method,
    headers: { "user-agent": userAgent, ...(request.headers ?? {}) },
    body,
    redirect: "follow",
    signal,
  });
  const result = {
    status: raw.status,
    url: raw.url || request.url,
    headers: Object.fromEntries(raw.headers.entries()),
    bytes: new Uint8Array(await raw.arrayBuffer()),
  };
  if (!challenged(result.status, result.url, result.headers, result.bytes)) return result;
  if (!browser.path) {
    state.needsBrowser = Date.now();
    return result; // the source sees the check and reports it, as in the app
  }
  routed.add(origin);
  try {
    return await browserFetch(request);
  } catch (error) {
    if (error instanceof ChallengeError) {
      state.challenge = { url: error.url, at: Date.now() };
      return result;
    }
    throw error;
  }
}

function createHttpBridge(state) {
  const bodies = new Map();
  const transfers = new Map();
  let nextId = 0;

  const store = (bytes) => {
    const bodyID = String(++nextId);
    bodies.set(bodyID, bytes);
    return { bodyID, sessionID: "urakatsu" };
  };

  const fetchNative = async (message) => {
    const controller = new AbortController();
    if (message.requestID) transfers.set(message.requestID, controller);
    const timer = setTimeout(() => controller.abort(), message.timeoutMs ?? 30_000);
    try {
      const request = {
        url: message.url,
        method: METHODS[message.method] ?? "GET",
        headers: message.headers ?? {},
        body: message.body,
        bytes: message.bodyKind === BODY_BYTES,
      };
      const result = await transportFetch(request, state, controller.signal);
      return { ...store(result.bytes), status: result.status, url: result.url, headers: result.headers };
    } finally {
      clearTimeout(timer);
      if (message.requestID) transfers.delete(message.requestID);
    }
  };

  return {
    now: () => Date.now(),
    async send(message) {
      switch (message.operation) {
        case Operation.Fetch:
          return fetchNative(message);
        case Operation.Read: {
          const bytes = bodies.get(message.bodyID);
          if (!bytes) throw new Error(`Response body ${message.bodyID} was released`);
          return { value: message.reader === READER_BASE64 ? Buffer.from(bytes).toString("base64") : decodeStrictUtf8(bytes) };
        }
        case Operation.Release:
          bodies.delete(message.bodyID);
          return {};
        case Operation.Cancel:
          transfers.get(message.requestID)?.abort();
          return {};
        case Operation.CreateBody:
          return store(new Uint8Array(Buffer.from(message.value, "base64")));
        default:
          return {};
      }
    },
  };
}

/** The app's key-value store, persisted to one JSON file per source. */
class ManaStore {
  constructor(file) {
    this.file = file;
    this.values = new Map();
    try {
      this.values = new Map(Object.entries(JSON.parse(fs.readFileSync(file, "utf-8"))));
    } catch {
      /* first use */
    }
  }

  save() {
    fs.mkdirSync(path.dirname(this.file), { recursive: true });
    fs.writeFileSync(this.file, JSON.stringify(Object.fromEntries(this.values)));
  }

  async get(k) {
    return this.values.has(k) ? this.values.get(k) : null;
  }

  async set(k, v) {
    this.values.set(k, v);
    this.save();
  }

  async remove(k) {
    this.values.delete(k);
    this.save();
  }

  typed(k, check, label) {
    const value = this.values.get(k);
    if (value === undefined) return null;
    if (!check(value)) throw new Error(`${k} is not a ${label}`);
    return value;
  }

  async string(k) {
    return this.typed(k, (v) => typeof v === "string", "string");
  }

  async boolean(k) {
    return this.typed(k, (v) => typeof v === "boolean", "boolean");
  }

  async number(k) {
    return this.typed(k, (v) => typeof v === "number", "number");
  }

  async stringArray(k) {
    return this.typed(k, (v) => Array.isArray(v) && v.every((e) => typeof e === "string"), "string array");
  }
}

/** The app's WebView for sources: a real browser page, one at a time per source. */
function createWebViewPage(state) {
  const WebViewPage = {
    active: false,
    async create() {
      if (WebViewPage.active) throw new Error("A WebView is already active for this source");
      WebViewPage.active = true;
      let view;
      try {
        view = await newView();
      } catch (error) {
        WebViewPage.active = false;
        if (error instanceof BrowserUnavailable) state.needsBrowser = Date.now();
        throw error;
      }
      let closed = false;
      return {
        async goto(url) {
          if (closed) throw new Error("WebView page is closed");
          browser.lastUse = Date.now();
          await view.navigate(url).catch((error) => log("navigate", url, error?.message));
          if (!(await settle(view))) {
            state.challenge = { url, at: Date.now() };
            throw new ChallengeError(url);
          }
        },
        async evaluateScript(script) {
          if (closed) throw new Error("WebView page is closed");
          // Indirect eval returns a script's completion value, as WKWebView's evaluateJavaScript does.
          return view.evaluate(`(0, eval)(${JSON.stringify(script)})`);
        },
        async close() {
          if (closed) return;
          closed = true;
          WebViewPage.active = false;
          view.close();
        },
      };
    },
  };
  return WebViewPage;
}

const quietConsole = { log, info: log, warn: log, error: log, debug: () => {} };

const targets = new Map(); // source -> {target, state}

async function load({ source, bundle, store }) {
  const code = fs.readFileSync(bundle, "utf-8");
  const state = { challenge: null, needsBrowser: 0 };
  const sandbox = {
    __manaHTTPBridge: createHttpBridge(state),
    ObjectStore: new ManaStore(store),
    SecureStore: new ManaStore(store.replace(/\.json$/, ".secure.json")),
    WebViewPage: createWebViewPage(state),
    console: quietConsole,
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    queueMicrotask,
    fetch,
    AbortController,
    TextEncoder,
    TextDecoder,
    URL,
    URLSearchParams,
    Buffer,
    atob,
    btoa,
    globalThis: undefined,
  };
  sandbox.globalThis = sandbox;
  sandbox.global = sandbox;
  sandbox.self = sandbox;
  const context = vm.createContext(sandbox);
  vm.runInContext(code, context, { filename: path.basename(bundle) });
  const Target = context.Target;
  if (!Target) throw new Error(`${path.basename(bundle)} exports no Target`);
  const target = new Target();
  // The app does not await this either; sources must not depend on it having finished.
  target.onEnvironmentLoaded?.()?.catch?.((error) => log(source, "onEnvironmentLoaded failed:", error?.message));
  targets.set(source, { target, state });
  return { info: plain(target.info ?? {}) };
}

function loaded(source) {
  const found = targets.get(source);
  if (!found) throw new Error(`source ${source} is not loaded`);
  return found;
}

const target = (source) => loaded(source).target;

/** Round-trips through JSON so values from the bundle's realm (Dates, arrays) serialize plainly. */
function plain(value) {
  return value === undefined ? null : JSON.parse(JSON.stringify(value));
}

function dateSeconds(value) {
  if (!value) return null;
  const time = typeof value.getTime === "function" ? value.getTime() : Date.parse(value);
  return Number.isFinite(time) && time > 0 ? time / 1000 : null;
}

function chapter(entry) {
  return {
    chapterId: String(entry.chapterId),
    title: entry.title ?? null,
    number: Number.isFinite(entry.number) ? entry.number : null,
    volume: Number.isFinite(entry.volume) ? entry.volume : null,
    index: Number.isFinite(entry.index) ? entry.index : null,
    language: entry.language ?? null,
    scanlator: (entry.providers ?? []).map((p) => p?.name ?? p).filter(Boolean).join(", ") || null,
    date: dateSeconds(entry.date),
  };
}

async function image(source, url, origin) {
  const { target: t, state } = loaded(source);
  const method = origin === ORIGIN_COVER ? "getCoverTransport" : "getPageTransport";
  const request = { url, origin };
  const client = typeof t[method] === "function" ? t[method](request) : undefined;
  if (client && typeof client[IMAGE_DISPATCH] === "function") {
    const response = await client[IMAGE_DISPATCH](request);
    response.requireOk();
    const bytes = await response.bytes();
    return { data: Buffer.from(bytes).toString("base64"), contentType: response.headers.get("content-type") ?? "" };
  }
  const result = await transportFetch({ url, method: "GET", headers: {} }, state);
  if (result.status >= 400) throw new Error(`HTTP ${result.status} for ${url}`);
  return { data: Buffer.from(result.bytes).toString("base64"), contentType: result.headers["content-type"] ?? "" };
}

// ---- interactive sessions -----------------------------------------------------------------------

function session(id) {
  const found = sessions.get(id);
  if (!found) throw new Error(`browser session ${id} is closed`);
  found.touched = Date.now();
  browser.lastUse = Date.now();
  return found;
}

const browserHandlers = {
  async "browser.configure"({ path: executable, profile }) {
    if (executable !== browser.path) {
      for (const origin of [...tabs.keys()]) closeTab(origin);
      browser.version = null;
    }
    browser.path = executable;
    browser.profile = profile;
    return {};
  },
  async "browser.status"() {
    return { path: browser.path, version: browser.version, tabs: [...tabs.keys()], routed: [...routed], sessions: sessions.size };
  },
  async "browser.open"({ url }) {
    const view = await newView();
    const id = String(++nextSession);
    sessions.set(id, { view, touched: Date.now() });
    // Page.navigate returns at once, so frames keep coming while the page loads.
    await view.cdp("Page.navigate", { url });
    return { id, width: VIEWPORT.width, height: VIEWPORT.height };
  },
  async "browser.frame"({ id }) {
    const { view } = session(id);
    const [image, state] = await Promise.all([view.screenshot({ format: "jpeg", quality: 70, encoding: "base64" }), pageState(view)]);
    return {
      image,
      title: state.title,
      url: state.url,
      loading: state.ready !== "complete",
      challenge: blockedPage(state),
      width: VIEWPORT.width,
      height: VIEWPORT.height,
    };
  },
  async "browser.input"({ id, action, x, y, text, key, dx, dy, url }) {
    const { view } = session(id);
    switch (action) {
      case "click":
        await view.click(x, y);
        break;
      case "type":
        await view.type(text);
        break;
      case "press":
        await view.press(key);
        break;
      case "scroll":
        await view.scroll(dx ?? 0, dy ?? 0);
        break;
      case "navigate":
        await view.cdp("Page.navigate", { url });
        break;
      case "back":
        await view.goBack();
        break;
      case "forward":
        await view.goForward();
        break;
      case "reload":
        await view.reload();
        break;
      default:
        throw new Error(`unknown browser action ${action}`);
    }
    return {};
  },
  async "browser.close"({ id }) {
    const found = sessions.get(id);
    sessions.delete(id);
    if (found) {
      const state = await pageState(found.view);
      found.view.close();
      // The person may have cleared a check: let that site's tab load again with the new cookies.
      try {
        const origin = new URL(state.url).origin;
        const tab = tabs.get(origin);
        if (tab) tab.cleared = false;
        routed.add(origin);
      } catch {
        /* no page loaded */
      }
    }
    for (const entry of targets.values()) entry.state.challenge = null;
    return {};
  },
};

setInterval(() => {
  const idle = Date.now() - browser.lastUse > BROWSER_IDLE_MS;
  for (const [id, found] of sessions) {
    if (Date.now() - found.touched > BROWSER_IDLE_MS) {
      found.view.close();
      sessions.delete(id);
    }
  }
  if (idle && sessions.size === 0 && tabs.size > 0) {
    for (const origin of [...tabs.keys()]) closeTab(origin);
    log("closed the idle browser");
  }
}, 60_000).unref?.();

// ---- protocol -----------------------------------------------------------------------------------

const handlers = {
  load,
  ...browserHandlers,
  async unload({ source }) {
    targets.delete(source);
    return {};
  },
  async search({ source, query, page }) {
    const result = await target(source).search({ query: query ?? "", page: page ?? 1 });
    return {
      results: plain(result?.results ?? []),
      isLastPage: result?.isLastPage ?? true,
    };
  },
  async content({ source, contentId }) {
    const content = await target(source).getContent(contentId);
    const chapters = Array.isArray(content?.chapters) ? content.chapters.map(chapter) : null;
    return { ...plain({ ...content, chapters: undefined }), chapters };
  },
  async chapters({ source, contentId }) {
    const t = target(source);
    const list = typeof t.getChapters === "function" ? await t.getChapters(contentId) : (await t.getContent(contentId))?.chapters;
    return (list ?? []).map(chapter);
  },
  async pages({ source, contentId, chapterId }) {
    const data = await target(source).getChapterData(contentId, chapterId);
    return (data?.pages ?? []).map((page) => (page.url ? { url: page.url } : { raw: page.raw }));
  },
  async image({ source, url, origin }) {
    return image(source, url, origin === "cover" ? ORIGIN_COVER : ORIGIN_PAGE);
  },
};

function reply(message) {
  process.stdout.write(JSON.stringify(message) + "\n");
}

/** What a failed call needs from Urakatsu: a person at the browser, or a browser at all. */
function blockers(source, error, started) {
  const state = targets.get(source)?.state;
  const recent = (at) => at && at >= started - 1000;
  const extra = {};
  if (error instanceof BrowserUnavailable || recent(state?.needsBrowser)) extra.needsBrowser = true;
  const challenge = error instanceof ChallengeError ? error.url : recent(state?.challenge?.at) ? state.challenge.url : null;
  if (challenge) extra.challenge = challenge;
  else if (error?.name === "CloudflareError" && error.resolutionURL && browser.path) extra.challenge = error.resolutionURL;
  return extra;
}

async function handle(line) {
  let request;
  try {
    request = JSON.parse(line);
  } catch {
    log("unreadable request:", line.slice(0, 200));
    return;
  }
  const handler = handlers[request.method];
  const started = Date.now();
  try {
    if (!handler) throw new Error(`unknown method ${request.method}`);
    reply({ id: request.id, result: await handler(request.params ?? {}) });
  } catch (error) {
    const name = error?.name && error.name !== "Error" ? `${error.name}: ` : "";
    reply({ id: request.id, error: name + String(error?.message ?? error), ...blockers(request.params?.source, error, started) });
  }
}

let buffered = "";
process.stdin.setEncoding("utf-8");
process.stdin.on("data", (chunk) => {
  buffered += chunk;
  let newline;
  while ((newline = buffered.indexOf("\n")) >= 0) {
    const line = buffered.slice(0, newline).trim();
    buffered = buffered.slice(newline + 1);
    if (line) void handle(line);
  }
});
process.stdin.on("end", () => {
  Bun.WebView.closeAll?.();
  process.exit(0);
});
reply({ id: 0, result: { ready: true, bun: typeof Bun !== "undefined" ? Bun.version : null } });
