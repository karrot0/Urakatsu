import { Slider as SliderPrimitive } from "@base-ui/react/slider"
import { cn } from "cn"

// The thumb is absolutely positioned by Base UI against the control, which must stay `relative`.
function Slider({ className, defaultValue, value, min = 0, max = 100, ...props }: SliderPrimitive.Root.Props) {
  const values = Array.isArray(value) ? value : Array.isArray(defaultValue) ? defaultValue : [value ?? defaultValue ?? min]

  return (
    <SliderPrimitive.Root
      className={cn("w-full touch-none", className)}
      data-slot="slider"
      defaultValue={defaultValue}
      value={value}
      min={min}
      max={max}
      thumbAlignment="edge"
      {...props}
    >
      <SliderPrimitive.Control className="relative flex h-4 w-full items-center select-none data-disabled:opacity-50">
        <SliderPrimitive.Track
          data-slot="slider-track"
          className="relative h-1 w-full rounded-full bg-surface-fill shadow-[inset_0_0_0_1px_var(--border-soft)] select-none"
        >
          <SliderPrimitive.Indicator data-slot="slider-range" className="rounded-[inherit] bg-brand select-none" />
        </SliderPrimitive.Track>
        {values.map((_, index) => (
          <SliderPrimitive.Thumb
            data-slot="slider-thumb"
            key={index}
            className="block size-3 shrink-0 rounded-full bg-surface shadow-[0_0_0_1px_var(--border-strong),0_1px_2px_rgba(25,25,25,0.12)] outline-none select-none focus-visible:shadow-[0_0_0_1px_var(--brand),0_0_0_4px_color-mix(in_oklab,var(--brand)_25%,transparent)] data-dragging:shadow-[0_0_0_1px_var(--brand),0_0_0_4px_color-mix(in_oklab,var(--brand)_25%,transparent)]"
          />
        ))}
      </SliderPrimitive.Control>
    </SliderPrimitive.Root>
  )
}

export { Slider }
