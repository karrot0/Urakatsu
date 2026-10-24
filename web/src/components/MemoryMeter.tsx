import { useEffect, useState } from "react";

import { catalog, formatBytes, type MemoryUsage } from "../api/catalog";
import { useWorkspace } from "../state/workspace";
import { Tooltip } from "./primitives";
import { UsageBar, UsageLegend, UsageRow } from "./UsageBar";

const POLL_MS = 3000;
/** A little shared GPU memory is normal (staging buffers); past this the driver is spilling VRAM. */
const SPILL_WARNING = 256 * 1024 ** 2;
const BACKENDS: Record<string, string> = { cuda: "CUDA", rocm: "ROCm", vulkan: "Vulkan", dml: "DirectML", metal: "Metal", cpu: "CPU" };

/** "cuda:0" as "CUDA:0", "rocm:0" as "ROCm:0"; the CPU has no index. */
function deviceLabel(label: string) {
  const [backend, index] = label.split(":");
  const name = BACKENDS[backend] ?? backend.toUpperCase();
  return backend === "cpu" || index === undefined ? name : `${name}:${index}`;
}

/**
 * GPU memory and system RAM, each as Urakatsu's share (brand), other apps (neutral) and free
 * space on a bar marked at each quarter. RAM holds model layers offloaded to the CPU. Polls while
 * the tab is visible.
 */
export function MemoryMeter() {
  const ws = useWorkspace();
  const [usage, setUsage] = useState<MemoryUsage | null>(null);

  useEffect(() => {
    let alive = true;
    let timer = 0;
    const poll = async () => {
      if (document.visibilityState === "visible") {
        try {
          const next = await catalog.memory();
          if (alive) setUsage(next);
        } catch {
          // A busy or older server; keep the last reading.
        }
      }
      if (alive) timer = window.setTimeout(poll, POLL_MS);
    };
    void poll();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, []);

  if (!usage?.ram) {
    const device = ws.capabilities?.device;
    if (!device) return null;
    return (
      // Distinct keys: Base UI binds hover listeners to the first trigger element it sees.
      <Tooltip key="device" content={device.description} side="top">
        <span className="ml-1 px-1 text-xs tabular-nums text-fg-2">{deviceLabel(`${device.backend}:${device.index}`)}</span>
      </Tooltip>
    );
  }

  const { gpu, ram } = usage;
  const spilling = !!gpu?.spilled && gpu.spilled >= SPILL_WARNING;
  return (
    <Tooltip
      key="memory"
      side="top"
      content={
        <div className="flex min-w-52 flex-col gap-2 py-0.5">
          {gpu && <UsageLegend pool={gpu} />}
          {gpu?.spilled ? (
            <span className={spilling ? "text-warning" : undefined}>
              <UsageRow label="Spilled to system memory" value={formatBytes(gpu.spilled)} />
              {spilling && <span className="opacity-70">VRAM is full, so the driver moved some of it to slower system memory.</span>}
            </span>
          ) : null}
          <UsageLegend pool={ram} note="Includes model layers offloaded to the CPU." />
        </div>
      }
    >
      <div className="ml-1 flex min-w-0 flex-1 cursor-default flex-col gap-1 px-1">
        {gpu && <UsageBar label={deviceLabel(usage.device)} pool={gpu} flag={spilling ? "Spilling" : undefined} />}
        <UsageBar label="RAM" pool={ram} />
      </div>
    </Tooltip>
  );
}
