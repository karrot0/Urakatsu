// Named formats, chosen once. Components never call toLocale* directly.

const dateTime = new Intl.DateTimeFormat(undefined, { day: "numeric", month: "short", year: "numeric", hour: "numeric", minute: "2-digit" });

/** Unix seconds as "23 Oct 2026, 10:42". */
export function formatTimestamp(seconds: number): string {
  return dateTime.format(new Date(seconds * 1000));
}

const day = new Intl.DateTimeFormat(undefined, { day: "numeric", month: "short", year: "numeric" });

/** Unix seconds as "23 Oct 2026". */
export function formatDay(seconds: number): string {
  return day.format(new Date(seconds * 1000));
}

/** An ISO date string as "23 Oct 2026". */
export function formatDate(iso: string): string {
  const parsed = new Date(iso);
  return Number.isNaN(parsed.getTime()) ? "" : day.format(parsed);
}

export function plural(count: number, one: string, many = `${one}s`): string {
  return `${count} ${count === 1 ? one : many}`;
}
