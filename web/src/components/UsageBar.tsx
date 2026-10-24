import { TriangleAlert } from "lucide-react";

import { formatBytes, type MemoryPool } from "../api/catalog";

export const APP_FILL = "bg-brand";
export const OTHER_FILL = "bg-[color-mix(in_oklab,var(--text-3)_45%,transparent)]";
export const FREE_FILL = "bg-surface-fill shadow-[inset_0_0_0_1px_var(--border)]";

/**
 * One pool (VRAM, RAM, a drive) as a bar marked at each quarter: Urakatsu's share in brand,
 * everything else in neutral, the rest free. The label line reads "share / total".
 */
export function UsageBar({ label, pool, flag, owner = "Urakatsu" }: { label: string; pool: MemoryPool; flag?: string; owner?: string }) {
  const app = pool.app ?? 0;
  const other = Math.max(0, pool.used - app);
  const share = (bytes: number) => `${pool.total ? Math.min(100, (bytes / pool.total) * 100) : 0}%`;
  return (
    <div
      className="flex flex-col gap-0.5"
      role="meter"
      aria-label={`${label} used by ${owner}`}
      aria-valuemin={0}
      aria-valuemax={pool.total}
      aria-valuenow={app}
      aria-valuetext={`${owner} ${formatBytes(app)}, ${formatBytes(pool.used)} of ${formatBytes(pool.total)} in use`}
    >
      <div className="flex items-baseline justify-between gap-2 text-xs leading-4 tabular-nums">
        <span className="flex flex-none items-baseline gap-1 text-fg-2">
          {label}
          {flag && (
            <span className="self-center text-warning">
              <TriangleAlert size={12} aria-hidden />
              <span className="sr-only">{flag}</span>
            </span>
          )}
        </span>
        <span className="truncate text-fg-3">
          <span className="text-fg-2">{formatBytes(pool.app ?? pool.used)}</span> / {formatBytes(pool.total)}
        </span>
      </div>
      <div className="relative flex h-1.5 overflow-hidden rounded-full bg-surface-fill">
        <span className={`h-full ${APP_FILL} transition-[width] duration-200 ease-out`} style={{ width: share(app) }} />
        <span className={`h-full ${OTHER_FILL} transition-[width] duration-200 ease-out`} style={{ width: share(other) }} />
        {[25, 50, 75].map((mark) => (
          <span key={mark} className="absolute inset-y-0 w-px bg-surface-sunk" style={{ left: `${mark}%` }} aria-hidden />
        ))}
      </div>
    </div>
  );
}

/** The bar's key, for a tooltip: the pool's name, then each part with its swatch. */
export function UsageLegend({ pool, note, owner = "Urakatsu", others = "Other apps" }: { pool: MemoryPool; note?: string; owner?: string; others?: string }) {
  const app = pool.app ?? 0;
  return (
    <div className="flex flex-col gap-1">
      <span className="font-medium">{pool.name}</span>
      <UsageRow swatch={APP_FILL} label={owner} value={pool.app === null ? "Not reported" : formatBytes(app)} />
      <UsageRow swatch={OTHER_FILL} label={others} value={formatBytes(Math.max(0, pool.used - app))} />
      <UsageRow swatch={FREE_FILL} label="Free" value={formatBytes(Math.max(0, pool.total - pool.used))} />
      {note && <span className="opacity-70">{note}</span>}
    </div>
  );
}

export function UsageRow({ swatch, label, value }: { swatch?: string; label: string; value: string }) {
  return (
    <span className="flex items-center gap-2 tabular-nums">
      {swatch && <span className={`size-2 flex-none rounded-full ${swatch}`} aria-hidden />}
      <span className="flex-1">{label}</span>
      <span>{value}</span>
    </span>
  );
}
