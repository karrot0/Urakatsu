// Keyboard shortcuts as small strings: "v", "0", "arrowleft", "mod+z", "mod+shift+z".
// "mod" is Ctrl, or Cmd on macOS. A shortcut without Ctrl/Cmd or Alt also fires with Shift
// held, which is how Shift+arrow nudges 10 pixels and Shift+V still picks the select tool.

export const IS_MAC = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);

const MODIFIER_KEYS = new Set(["control", "shift", "alt", "meta", "os", "altgraph"]);

interface Parsed {
  mod: boolean;
  shift: boolean;
  alt: boolean;
  key: string;
}

function parse(binding: string): Parsed {
  const parts = binding.toLowerCase().split("+");
  // "mod++" is Ctrl and the plus key.
  const key = binding.endsWith("++") ? "+" : (parts.pop() ?? "");
  return { mod: parts.includes("mod"), shift: parts.includes("shift"), alt: parts.includes("alt"), key };
}

function eventKey(event: KeyboardEvent): string {
  return event.key === " " ? "space" : event.key.toLowerCase();
}

/** Whether ``event`` is the shortcut ``binding``. */
export function matches(binding: string | undefined, event: KeyboardEvent): boolean {
  if (!binding) return false;
  const wanted = parse(binding);
  const mod = event.ctrlKey || event.metaKey;
  if (wanted.key !== eventKey(event) || wanted.mod !== mod || wanted.alt !== event.altKey) return false;
  return wanted.mod || wanted.alt || wanted.shift ? wanted.shift === event.shiftKey : true;
}

/** The shortcut a key press records, or null for a lone modifier key. */
export function fromEvent(event: KeyboardEvent | React.KeyboardEvent): string | null {
  const key = event.key === " " ? "space" : event.key.toLowerCase();
  if (MODIFIER_KEYS.has(key)) return null;
  const parts = [];
  if (event.ctrlKey || event.metaKey) parts.push("mod");
  if (event.altKey) parts.push("alt");
  // Shift only counts with another modifier: Shift+letter is the letter (see the note above).
  if (event.shiftKey && parts.length > 0) parts.push("shift");
  parts.push(key);
  return parts.join("+");
}

const KEY_NAMES: Record<string, string> = {
  arrowleft: "←",
  arrowright: "→",
  arrowup: "↑",
  arrowdown: "↓",
  space: "Space",
  escape: "Esc",
  delete: "Del",
  backspace: "Backspace",
  enter: "Enter",
  tab: "Tab",
};

/** "Ctrl+Shift+Z" (or "⌘⇧Z" on macOS) for display. */
export function format(binding: string | undefined): string {
  if (!binding) return "";
  const parsed = parse(binding);
  const key = KEY_NAMES[parsed.key] ?? (parsed.key.length === 1 ? parsed.key.toUpperCase() : parsed.key.charAt(0).toUpperCase() + parsed.key.slice(1));
  if (IS_MAC) return `${parsed.mod ? "⌘" : ""}${parsed.alt ? "⌥" : ""}${parsed.shift ? "⇧" : ""}${key}`;
  return [parsed.mod && "Ctrl", parsed.alt && "Alt", parsed.shift && "Shift", key].filter(Boolean).join("+");
}

/** The name of the platform's main modifier, for help text. */
export const MOD_NAME = IS_MAC ? "⌘" : "Ctrl";
