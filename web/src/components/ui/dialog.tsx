import * as React from "react"
import { Dialog as DialogPrimitive } from "@base-ui/react/dialog"
import { cn } from "cn"
import { XIcon } from "lucide-react"

import { Button } from "@/components/ui/button"

function Dialog({ ...props }: DialogPrimitive.Root.Props) {
  return <DialogPrimitive.Root data-slot="dialog" {...props} />
}

function DialogTrigger({ ...props }: DialogPrimitive.Trigger.Props) {
  return <DialogPrimitive.Trigger data-slot="dialog-trigger" {...props} />
}

function DialogPortal({ ...props }: DialogPrimitive.Portal.Props) {
  return <DialogPrimitive.Portal data-slot="dialog-portal" {...props} />
}

function DialogClose({ ...props }: DialogPrimitive.Close.Props) {
  return <DialogPrimitive.Close data-slot="dialog-close" {...props} />
}

function DialogOverlay({ className, ...props }: DialogPrimitive.Backdrop.Props) {
  return (
    <DialogPrimitive.Backdrop
      data-slot="dialog-overlay"
      className={cn(
        "fixed inset-0 z-50 bg-[color-mix(in_oklab,var(--n-950)_40%,transparent)] transition-opacity duration-150 data-ending-style:opacity-0 data-starting-style:opacity-0 max-md:bg-glass-scrim max-md:duration-300",
        className
      )}
      {...props}
    />
  )
}

/**
 * Modal surface: the largest radius, the modal shadow plus a hairline, centered. On a phone it
 * is a glass sheet that rises from the bottom, clear of the home indicator.
 */
function DialogContent({
  className,
  children,
  showCloseButton = true,
  ...props
}: DialogPrimitive.Popup.Props & { showCloseButton?: boolean }) {
  return (
    <DialogPortal>
      <DialogOverlay />
      <DialogPrimitive.Popup
        data-slot="dialog-content"
        className={cn(
          "fixed top-1/2 left-1/2 z-[51] flex w-full max-w-[calc(100%-2rem)] -translate-x-1/2 -translate-y-1/2 flex-col gap-4 rounded-modal bg-surface p-5 text-sm text-fg shadow-[var(--shadow-modal),0_0_0_1px_var(--border-soft)] outline-none transition-opacity duration-150 data-ending-style:opacity-0 data-starting-style:opacity-0 sm:max-w-md",
          "max-md:glass-thick max-md:top-auto max-md:right-2 max-md:bottom-2 max-md:left-2 max-md:max-h-[calc(100dvh_-_var(--safe-top)_-_24px)] max-md:w-auto max-md:max-w-none max-md:translate-x-0 max-md:translate-y-0 max-md:overflow-y-auto max-md:rounded-sheet max-md:pb-[calc(var(--safe-bottom)+20px)] max-md:text-[15px] max-md:transition-[translate,opacity] max-md:duration-300 max-md:ease-sheet max-md:data-ending-style:translate-y-[110%] max-md:data-ending-style:opacity-100 max-md:data-starting-style:translate-y-[110%] max-md:data-starting-style:opacity-100",
          className
        )}
        {...props}
      >
        {children}
        {showCloseButton && (
          <DialogPrimitive.Close
            data-slot="dialog-close"
            render={<Button variant="ghost" className="absolute top-3 right-3" size="icon-sm" />}
          >
            <XIcon size={16} />
            <span className="sr-only">Close</span>
          </DialogPrimitive.Close>
        )}
      </DialogPrimitive.Popup>
    </DialogPortal>
  )
}

function DialogHeader({ className, ...props }: React.ComponentProps<"div">) {
  return <div data-slot="dialog-header" className={cn("flex flex-col gap-1", className)} {...props} />
}

function DialogFooter({ className, ...props }: React.ComponentProps<"div">) {
  return <div data-slot="dialog-footer" className={cn("flex justify-end gap-2", className)} {...props} />
}

function DialogTitle({ className, ...props }: DialogPrimitive.Title.Props) {
  return <DialogPrimitive.Title data-slot="dialog-title" className={cn("text-base font-semibold", className)} {...props} />
}

function DialogDescription({ className, ...props }: DialogPrimitive.Description.Props) {
  return <DialogPrimitive.Description data-slot="dialog-description" className={cn("text-sm text-fg-2", className)} {...props} />
}

export { Dialog, DialogClose, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogOverlay, DialogPortal, DialogTitle, DialogTrigger }
