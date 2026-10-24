import type { Ref } from "react"
import { ScrollArea as ScrollAreaPrimitive } from "@base-ui/react/scroll-area"
import { cn } from "cn"

type ScrollAreaProps = ScrollAreaPrimitive.Root.Props & {
  /** Which scrollbars to draw. A vertical-only area lets its content shrink to the viewport, so rows still truncate. */
  scrollbars?: "vertical" | "horizontal" | "both"
  viewportRef?: Ref<HTMLDivElement>
  viewportClassName?: string
  contentClassName?: string
}

function ScrollArea({
  className,
  children,
  scrollbars = "vertical",
  viewportRef,
  viewportClassName,
  contentClassName,
  ...props
}: ScrollAreaProps) {
  return (
    <ScrollAreaPrimitive.Root
      data-slot="scroll-area"
      className={cn("relative min-h-0 overflow-hidden", className)}
      {...props}
    >
      <ScrollAreaPrimitive.Viewport
        ref={viewportRef}
        data-slot="scroll-area-viewport"
        className={cn("size-full rounded-[inherit] outline-none focus-visible:shadow-[inset_0_0_0_2px_var(--brand)]", viewportClassName)}
      >
        <ScrollAreaPrimitive.Content
          data-slot="scroll-area-content"
          className={contentClassName}
          style={scrollbars === "vertical" ? { minWidth: 0 } : undefined}
        >
          {children}
        </ScrollAreaPrimitive.Content>
      </ScrollAreaPrimitive.Viewport>
      {scrollbars !== "horizontal" && <ScrollBar />}
      {scrollbars !== "vertical" && <ScrollBar orientation="horizontal" />}
      <ScrollAreaPrimitive.Corner />
    </ScrollAreaPrimitive.Root>
  )
}

/** Hidden until the pointer is over the area or it scrolls; fades out slower than it fades in. */
function ScrollBar({
  className,
  orientation = "vertical",
  ...props
}: ScrollAreaPrimitive.Scrollbar.Props) {
  return (
    <ScrollAreaPrimitive.Scrollbar
      data-slot="scroll-area-scrollbar"
      orientation={orientation}
      className={cn(
        "z-10 flex touch-none p-0.5 opacity-0 transition-opacity duration-200 ease-out select-none",
        "data-hovering:opacity-100 data-hovering:duration-150 data-scrolling:opacity-100 data-scrolling:duration-0",
        "data-[orientation=horizontal]:h-2.5 data-[orientation=horizontal]:flex-col data-[orientation=vertical]:w-2.5",
        className
      )}
      {...props}
    >
      <ScrollAreaPrimitive.Thumb
        data-slot="scroll-area-thumb"
        className="relative flex-1 rounded-full bg-[color-mix(in_oklab,var(--text-3)_45%,transparent)] transition-colors duration-[20ms] ease-in hover:bg-[color-mix(in_oklab,var(--text-3)_75%,transparent)] active:bg-[color-mix(in_oklab,var(--text-2)_75%,transparent)]"
      />
    </ScrollAreaPrimitive.Scrollbar>
  )
}

export { ScrollArea, ScrollBar }
