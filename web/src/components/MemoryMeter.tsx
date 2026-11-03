import { useEffect, useState } from "react";

import { STAGE_LABEL, catalog, formatBytes, type MemoryUsage } from "../api/catalog";
import { describeError } from "../api/client";
import { useWorkspace } from "../state/workspace";
import { Popover, Tooltip } from "./primitives";
import { UsageBar, UsageLegend, UsageRow } from "./UsageBar";
import { Button } from "@/components/ui/button";

const POLL_MS = 3000;
const ALL = "*"; // unloading every model
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
 * the tab is visible. A click opens the details, with the loaded models and a way to unload them.
 */
export function MemoryMeter() {
  const ws = useWorkspace();
  const [usage, setUsage] = useState<MemoryUsage | null>(null);
  const [unloading, setUnloading] = useState<string>();

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
  const unload = async (stage?: string) => {
    setUnloading(stage ?? ALL);
    try {
      await catalog.unloadModels(stage);
    } catch (error) {
      ws.reportError(describeError(error));
    } finally {
      setUnloading(undefined);
      catalog.memory().then(setUsage, () => undefined);
    }
  };
  return (
    <Popover
      side="top"
      align="end"
      className="w-72"
      trigger={
        <button type="button" aria-label="Memory and loaded models" className="ml-1 flex min-w-0 flex-1 cursor-pointer flex-col gap-1 rounded-control px-1 py-0.5 text-left transition-colors duration-[20ms] ease-in hover:bg-hover focus-visible:shadow-focus focus-visible:outline-none data-[popup-open]:bg-hover">
          {gpu && <UsageBar label={deviceLabel(usage.device)} pool={gpu} flag={spilling ? "Spilling" : undefined} />}
          <UsageBar label="RAM" pool={ram} />
        </button>
      }
    >
      <div className="flex flex-col gap-2 p-1">
        {gpu && <UsageLegend pool={gpu} />}
        {gpu?.spilled ? (
          <span className={spilling ? "text-warning" : undefined}>
            <UsageRow label="Spilled to system memory" value={formatBytes(gpu.spilled)} />
            {spilling && <span className="opacity-70">VRAM is full, so the driver moved some of it to slower system memory.</span>}
          </span>
        ) : null}
        <UsageLegend pool={ram} note="Includes model layers offloaded to the CPU." />
        {usage.loaded && (
          <div className="flex flex-col gap-1 border-t border-line-soft pt-2">
            <div className="flex items-center justify-between gap-2">
              <span className="font-medium">Loaded models</span>
              {usage.loaded.length > 1 && (
                <Button variant="ghost" size="xs" disabled={usage.busy || !!unloading} onClick={() => void unload()}>
                  {unloading === ALL ? "Unloading" : "Unload all"}
                </Button>
              )}
            </div>
            {usage.loaded.length === 0 && <span className="text-xs text-fg-2">None. A run loads what it needs.</span>}
            {usage.loaded.map((model) => (
              <div key={model.stage} className="flex min-h-7 items-center justify-between gap-2">
                <span className="min-w-0 truncate" title={model.provider}>
                  {model.name}
                  <span className="text-xs text-fg-3"> {STAGE_LABEL[model.stage] ?? model.stage}</span>
                </span>
                <Button variant="outline" size="xs" disabled={usage.busy || !!unloading} onClick={() => void unload(model.stage)}>
                  {unloading === model.stage || unloading === ALL ? "Unloading" : "Unload"}
                </Button>
              </div>
            ))}
            {usage.busy && usage.loaded.length > 0 && <span className="text-xs text-fg-2">A run is using the models; they can be unloaded once it ends.</span>}
          </div>
        )}
      </div>
    </Popover>
  );
}
